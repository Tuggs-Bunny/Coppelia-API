from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional
import asyncpg
import asyncio
import os
import json
import math
import logging
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_URL = f"postgresql://postgres:{DB_PASSWORD}@localhost/coppelia"

pool = None

async def get_pool():
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(DB_URL, min_size=5, max_size=20)
    return pool

@app.on_event("startup")
async def startup():
    p = await get_pool()
    async with p.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS listening_events (
                id SERIAL PRIMARY KEY,
                hashed_user_id VARCHAR(64) NOT NULL,
                track_name VARCHAR(255) NOT NULL,
                artist_name VARCHAR(255) NOT NULL,
                genre VARCHAR(100) NOT NULL DEFAULT 'Unknown',
                completion_percentage FLOAT NOT NULL DEFAULT 0.0,
                was_skipped BOOLEAN NOT NULL DEFAULT FALSE,
                was_loved BOOLEAN NOT NULL DEFAULT FALSE,
                was_replayed BOOLEAN NOT NULL DEFAULT FALSE,
                time_of_day VARCHAR(20) NOT NULL DEFAULT 'unknown',
                listened_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                id SERIAL PRIMARY KEY,
                track_name VARCHAR(255) NOT NULL,
                artist_name VARCHAR(255) NOT NULL,
                genre VARCHAR(100),
                first_seen TIMESTAMP DEFAULT NOW(),
                play_count INTEGER DEFAULT 1,
                UNIQUE(track_name, artist_name)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                hashed_user_id VARCHAR(64) PRIMARY KEY,
                genre_weights JSONB NOT NULL DEFAULT '{}',
                artist_weights JSONB NOT NULL DEFAULT '{}',
                time_weights JSONB NOT NULL DEFAULT '{}',
                total_events INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_similarity (
                user_a VARCHAR(64) NOT NULL,
                user_b VARCHAR(64) NOT NULL,
                similarity_score FLOAT NOT NULL,
                calculated_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (user_a, user_b)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS opt_outs (
                hashed_user_id VARCHAR(64) PRIMARY KEY,
                opted_out_at TIMESTAMP DEFAULT NOW()
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_listening_user ON listening_events(hashed_user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_listening_track ON listening_events(track_name, artist_name)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_similarity_user_a ON user_similarity(user_a, similarity_score DESC)")
    log.info("Database ready")
    asyncio.create_task(similarity_job())

def clamp(value, min_val=-1.0, max_val=1.0):
    return max(min_val, min(max_val, value))

def calculate_weight_delta(completion: float, skipped: bool, loved: bool, replayed: bool) -> float:
    if loved:
        return 0.5
    if replayed:
        return 0.4
    if completion > 0.8:
        return 0.3
    if completion > 0.45 and skipped:
        return 0.1
    if skipped and completion < 0.45:
        return -0.1
    return 0.1

def update_weights(weights: dict, key: str, delta: float) -> dict:
    weights = dict(weights)
    weights[key] = clamp((weights.get(key, 0.0) or 0.0) + delta)
    return weights

class ListenEvent(BaseModel):
    hashed_user_id: str
    track_name: str
    artist_name: str
    genre: str = "Unknown"
    completion_percentage: float = 0.0
    was_skipped: bool = False
    was_loved: bool = False
    was_replayed: bool = False
    time_of_day: str = "unknown"

@app.post("/listen")
@limiter.limit("60/minute")
async def record_listen(request: Request, event: ListenEvent):
    try:
        p = await get_pool()
        async with p.acquire() as conn:
            await conn.execute("""
                INSERT INTO listening_events
                (hashed_user_id, track_name, artist_name, genre, completion_percentage,
                 was_skipped, was_loved, was_replayed, time_of_day)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            """, event.hashed_user_id, event.track_name, event.artist_name,
                event.genre, event.completion_percentage, event.was_skipped,
                event.was_loved, event.was_replayed, event.time_of_day)

            await conn.execute("""
                INSERT INTO tracks (track_name, artist_name, genre)
                VALUES ($1, $2, $3)
                ON CONFLICT (track_name, artist_name)
                DO UPDATE SET play_count = tracks.play_count + 1
            """, event.track_name, event.artist_name, event.genre)

            profile = await conn.fetchrow(
                "SELECT genre_weights, artist_weights, time_weights FROM user_profiles WHERE hashed_user_id = $1",
                event.hashed_user_id
            )

            delta = calculate_weight_delta(
                event.completion_percentage, event.was_skipped,
                event.was_loved, event.was_replayed
            )

            if profile:
                gw = update_weights(json.loads(profile['genre_weights'] or '{}'), event.genre, delta)
                aw = update_weights(json.loads(profile['artist_weights'] or '{}'), event.artist_name, delta)
                tw = update_weights(json.loads(profile['time_weights'] or '{}'), event.time_of_day, delta)
                await conn.execute("""
                    UPDATE user_profiles
                    SET genre_weights=$1, artist_weights=$2, time_weights=$3,
                        total_events=total_events+1, updated_at=NOW()
                    WHERE hashed_user_id=$4
                """, json.dumps(gw), json.dumps(aw), json.dumps(tw), event.hashed_user_id)
            else:
                gw = {event.genre: clamp(delta)}
                aw = {event.artist_name: clamp(delta)}
                tw = {event.time_of_day: clamp(delta)}
                await conn.execute("""
                    INSERT INTO user_profiles (hashed_user_id, genre_weights, artist_weights, time_weights, total_events)
                    VALUES ($1,$2,$3,$4,1)
                """, event.hashed_user_id, json.dumps(gw), json.dumps(aw), json.dumps(tw))

        return {"status": "ok"}
    except Exception as e:
        log.error(f"Error in record_listen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations")
async def get_recommendations(id: str, limit: int = 50):
    try:
        p = await get_pool()
        async with p.acquire() as conn:
            opted_out = await conn.fetchrow("SELECT 1 FROM opt_outs WHERE hashed_user_id=$1", id)
            if opted_out:
                return []

            similar_users = await conn.fetch("""
                SELECT user_b, similarity_score FROM user_similarity
                WHERE user_a=$1 ORDER BY similarity_score DESC LIMIT 20
            """, id)

            if len(similar_users) < 3:
                rows = await conn.fetch("""
                    SELECT t.track_name, t.artist_name, t.genre, t.play_count::float as score
                    FROM tracks t
                    WHERE (t.track_name, t.artist_name) NOT IN (
                        SELECT track_name, artist_name FROM listening_events WHERE hashed_user_id=$1
                    )
                    ORDER BY t.play_count DESC LIMIT $2
                """, id, limit)
                return [{"track": r['track_name'], "artist": r['artist_name'], "genre": r['genre'], "score": r['score']} for r in rows]

            similar_ids = [r['user_b'] for r in similar_users]
            sim_map = {r['user_b']: r['similarity_score'] for r in similar_users}

            candidates = await conn.fetch("""
                SELECT track_name, artist_name, genre, completion_percentage, was_loved, hashed_user_id
                FROM listening_events
                WHERE hashed_user_id = ANY($1::varchar[])
                AND (completion_percentage > 0.7 OR was_loved = true)
                AND (track_name, artist_name) NOT IN (
                    SELECT track_name, artist_name FROM listening_events WHERE hashed_user_id=$2
                )
            """, similar_ids, id)

            scores = {}
            for row in candidates:
                key = (row['track_name'], row['artist_name'], row['genre'])
                sim = sim_map.get(row['hashed_user_id'], 0.0)
                base = sim * row['completion_percentage']
                if row['was_loved']:
                    base += 0.2
                scores[key] = scores.get(key, 0.0) + base

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            return [{"track": k[0], "artist": k[1], "genre": k[2], "score": round(v, 4)} for k, v in ranked]

    except Exception as e:
        log.error(f"Error in get_recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tracks")
async def get_tracks(genre: Optional[str] = None, limit: int = 50):
    try:
        p = await get_pool()
        async with p.acquire() as conn:
            if genre:
                rows = await conn.fetch(
                    "SELECT track_name, artist_name, genre, play_count FROM tracks WHERE genre=$1 ORDER BY play_count DESC LIMIT $2",
                    genre, limit
                )
            else:
                rows = await conn.fetch(
                    "SELECT track_name, artist_name, genre, play_count FROM tracks ORDER BY play_count DESC LIMIT $1",
                    limit
                )
            return [{"track": r['track_name'], "artist": r['artist_name'], "genre": r['genre'], "play_count": r['play_count']} for r in rows]
    except Exception as e:
        log.error(f"Error in get_tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/user/{hashed_user_id}")
async def delete_user(hashed_user_id: str):
    try:
        p = await get_pool()
        async with p.acquire() as conn:
            await conn.execute("DELETE FROM listening_events WHERE hashed_user_id=$1", hashed_user_id)
            await conn.execute("DELETE FROM user_profiles WHERE hashed_user_id=$1", hashed_user_id)
            await conn.execute("DELETE FROM user_similarity WHERE user_a=$1 OR user_b=$1", hashed_user_id)
            await conn.execute("INSERT INTO opt_outs (hashed_user_id) VALUES ($1) ON CONFLICT DO NOTHING", hashed_user_id)
        return {"status": "deleted"}
    except Exception as e:
        log.error(f"Error in delete_user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    try:
        p = await get_pool()
        async with p.acquire() as conn:
            track_count = await conn.fetchval("SELECT COUNT(*) FROM tracks")
            user_count = await conn.fetchval("SELECT COUNT(*) FROM user_profiles")
        return {"status": "ok", "time": datetime.now().isoformat(), "total_tracks": track_count, "total_users": user_count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def cosine_similarity(a: dict, b: dict) -> float:
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    dot = sum((a.get(k, 0.0) or 0.0) * (b.get(k, 0.0) or 0.0) for k in keys)
    mag_a = math.sqrt(sum((a.get(k, 0.0) or 0.0) ** 2 for k in keys))
    mag_b = math.sqrt(sum((b.get(k, 0.0) or 0.0) ** 2 for k in keys))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

async def recalculate_similarity():
    try:
        p = await get_pool()
        async with p.acquire() as conn:
            profiles = await conn.fetch(
                "SELECT hashed_user_id, genre_weights, artist_weights FROM user_profiles WHERE total_events >= 5"
            )
            if len(profiles) < 2:
                return
            parsed = []
            for p_row in profiles:
                gw = json.loads(p_row['genre_weights'] or '{}')
                aw = json.loads(p_row['artist_weights'] or '{}')
                combined = {f"g_{k}": v for k, v in gw.items()}
                combined.update({f"a_{k}": v for k, v in aw.items()})
                parsed.append((p_row['hashed_user_id'], combined))

            pairs = []
            for i in range(len(parsed)):
                for j in range(i + 1, len(parsed)):
                    uid_a, vec_a = parsed[i]
                    uid_b, vec_b = parsed[j]
                    score = cosine_similarity(vec_a, vec_b)
                    if score > 0.3:
                        pairs.append((uid_a, uid_b, score))
                        pairs.append((uid_b, uid_a, score))

            async with p.acquire() as conn2:
                for uid_a, uid_b, score in pairs:
                    await conn2.execute("""
                        INSERT INTO user_similarity (user_a, user_b, similarity_score)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (user_a, user_b) DO UPDATE SET similarity_score=$3, calculated_at=NOW()
                    """, uid_a, uid_b, score)

        log.info(f"Similarity recalculated: {len(pairs)} pairs")
    except Exception as e:
        log.error(f"Similarity job error: {e}")

async def similarity_job():
    while True:
        await asyncio.sleep(3600)
        await recalculate_similarity()