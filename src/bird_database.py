#!/usr/bin/env python3
"""SQLite database for bird sightings."""
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BirdDatabase:
    """SQLite database for tracking bird sightings."""

    def __init__(self, db_path: Path):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row  # Return rows as dicts
        self.cursor = self.conn.cursor()

        self._create_tables()
        logger.info(f"Database initialized: {db_path}")

    def _create_tables(self):
        """Create database tables if they don't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                species TEXT NOT NULL,
                common_name TEXT,
                confidence REAL NOT NULL,
                frame_width INTEGER,
                frame_height INTEGER,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                stream_source TEXT,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON sightings(timestamp)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_species ON sightings(species)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence ON sightings(confidence)
        """)

        self.conn.commit()

    def add_sighting(self, species: str, confidence: float,
                     bbox: Optional[tuple] = None,
                     frame_size: Optional[tuple] = None,
                     stream_source: str = "unknown",
                     notes: Optional[str] = None,
                     timestamp: Optional[datetime] = None) -> int:
        """
        Add a bird sighting to the database.

        Args:
            species: Full species name (e.g., "Cardinalis cardinalis (Northern Cardinal)")
            confidence: Classification confidence (0.0-1.0)
            bbox: Bounding box (x, y, w, h) or None
            frame_size: Frame dimensions (width, height) or None
            stream_source: Stream identifier (e.g., "substream", "mainstream")
            notes: Optional notes about sighting
            timestamp: Sighting timestamp (default: now)

        Returns:
            Sighting ID
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Extract common name from species string
        common_name = None
        if '(' in species and ')' in species:
            common_name = species.split('(')[1].split(')')[0]

        # Unpack bbox if provided
        bbox_x, bbox_y, bbox_w, bbox_h = bbox if bbox else (None, None, None, None)

        # Unpack frame size if provided
        frame_width, frame_height = frame_size if frame_size else (None, None)

        self.cursor.execute("""
            INSERT INTO sightings
            (timestamp, species, common_name, confidence,
             frame_width, frame_height, bbox_x, bbox_y, bbox_w, bbox_h,
             stream_source, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, species, common_name, confidence,
              frame_width, frame_height, bbox_x, bbox_y, bbox_w, bbox_h,
              stream_source, notes))

        self.conn.commit()
        sighting_id = self.cursor.lastrowid

        logger.info(f"Sighting recorded: {common_name or species} @ {confidence:.1%} (ID: {sighting_id})")
        return sighting_id

    def get_recent_sightings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent sightings."""
        self.cursor.execute("""
            SELECT * FROM sightings
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in self.cursor.fetchall()]

    def get_species_stats(self) -> List[Dict[str, Any]]:
        """Get statistics per species."""
        self.cursor.execute("""
            SELECT
                common_name,
                species,
                COUNT(*) as visit_count,
                AVG(confidence) as avg_confidence,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM sightings
            WHERE species != 'background'
            GROUP BY species
            ORDER BY visit_count DESC
        """)

        return [dict(row) for row in self.cursor.fetchall()]

    def get_daily_activity(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily bird activity for last N days."""
        self.cursor.execute("""
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as total_sightings,
                COUNT(DISTINCT species) as unique_species
            FROM sightings
            WHERE timestamp >= DATE('now', ? || ' days')
              AND species != 'background'
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """, (f'-{days}',))

        return [dict(row) for row in self.cursor.fetchall()]

    def filter_background(self, threshold: float = 0.5):
        """Filter out low-confidence background detections."""
        self.cursor.execute("""
            DELETE FROM sightings
            WHERE species = 'background' AND confidence > ?
        """, (threshold,))

        deleted = self.cursor.rowcount
        self.conn.commit()

        logger.info(f"Deleted {deleted} background detections")
        return deleted

    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("Database connection closed")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


def main():
    """Test database functionality."""
    import sys

    db_path = Path("/volume1/docker/bird-id/data/birds.db")

    print(f"Testing BirdDatabase: {db_path}")

    with BirdDatabase(db_path) as db:
        # Add test sighting
        sighting_id = db.add_sighting(
            species="Cardinalis cardinalis (Northern Cardinal)",
            confidence=0.92,
            bbox=(100, 200, 50, 60),
            frame_size=(1920, 1080),
            stream_source="mainstream",
            notes="Test sighting"
        )
        print(f"\nAdded sighting ID: {sighting_id}")

        # Get recent sightings
        print("\nRecent sightings:")
        recent = db.get_recent_sightings(limit=5)
        for sighting in recent:
            print(f"  {sighting['timestamp']}: {sighting['common_name']} ({sighting['confidence']:.1%})")

        # Get species stats
        print("\nSpecies statistics:")
        stats = db.get_species_stats()
        for stat in stats:
            print(f"  {stat['common_name']}: {stat['visit_count']} visits, avg {stat['avg_confidence']:.1%}")

    print("\nDatabase test complete!")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
