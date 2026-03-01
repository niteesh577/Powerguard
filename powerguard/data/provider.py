"""
data/provider.py
================
Abstract data provider Protocol.

Phase 1: MockDataProvider (this file's companion) returns hardcoded data.
Phase 2: Replace with SupabaseDataProvider — same interface, no agent changes needed.

Dependency injection in api/routes/analysis.py:
    provider = MockDataProvider()          # Phase 1
    # provider = SupabaseDataProvider(db)  # Phase 2 — one-line swap
"""

from typing import Protocol, runtime_checkable

from .models import UserProfile, WorkoutLog, NutritionLog, SleepLog


@runtime_checkable
class UserDataProvider(Protocol):
    """
    Interface contract for all user data sources.
    Any class that implements these 4 methods is a valid provider.
    """

    async def get_user_profile(self, user_id: str) -> UserProfile:
        """Fetch basic user biodata and maxes."""
        ...

    async def get_workout_logs(self, user_id: str, days: int = 28) -> list[WorkoutLog]:
        """
        Fetch workout history for the last `days` days.
        Returns empty list if no data found (never raises for missing user).
        """
        ...

    async def get_nutrition_logs(self, user_id: str, days: int = 7) -> list[NutritionLog]:
        """Fetch nutrition logs for the last `days` days."""
        ...

    async def get_sleep_logs(self, user_id: str, days: int = 7) -> list[SleepLog]:
        """Fetch sleep + recovery logs for the last `days` days."""
        ...
