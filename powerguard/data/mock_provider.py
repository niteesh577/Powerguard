"""
data/mock_provider.py
=====================
MockDataProvider — returns realistic hardcoded data for any user_id.

FUTURE: Replace with SupabaseDataProvider:

    class SupabaseDataProvider:
        def __init__(self, supabase_client):
            self.db = supabase_client

        async def get_user_profile(self, user_id: str) -> UserProfile:
            result = await self.db.table("user_profiles").select("*").eq("user_id", user_id).execute()
            return UserProfile(**result.data[0])

        async def get_workout_logs(self, user_id: str, days: int = 28) -> list[WorkoutLog]:
            cutoff = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
            result = await self.db.table("workout_logs") \\
                .select("*").eq("user_id", user_id).gte("date", cutoff).execute()
            return [WorkoutLog(**r) for r in result.data]

        # ... same pattern for nutrition_logs and sleep_logs
"""

from datetime import date, timedelta

from .models import (
    ExperienceLevel, UserProfile,
    WorkoutLog, NutritionLog, SleepLog,
)


def _days_ago(n: int) -> date:
    return date.today() - timedelta(days=n)


class MockDataProvider:
    """
    Realistic mock data for development and testing.
    All methods are async to match the SupabaseDataProvider interface.
    user_id is accepted but ignored — same data is returned for any user.
    """

    async def get_user_profile(self, user_id: str) -> UserProfile:
        return UserProfile(
            user_id=user_id,
            name="Alex Johnson",
            age=28,
            sex="male",
            bodyweight_kg=85.0,
            training_age_years=4.0,
            experience_level=ExperienceLevel.intermediate,
            injury_history=["left knee strain (2022)", "lower back strain (2023)"],
            max_bench_kg=122.5,
            max_deadlift_kg=185.0,
            max_squat_kg=152.5,
        )

    async def get_workout_logs(self, user_id: str, days: int = 28) -> list[WorkoutLog]:
        # 28 days of realistic powerlifting logs (4-day/week programme)
        return [
            # Week 4 (most recent)
            WorkoutLog(date=_days_ago(1),  exercise="bench_press", weight_kg=102.5, reps=4, sets=4, rpe=8.5, velocity_ms=0.42),
            WorkoutLog(date=_days_ago(2),  exercise="deadlift",    weight_kg=170.0, reps=3, sets=4, rpe=9.0, velocity_ms=0.35),
            WorkoutLog(date=_days_ago(4),  exercise="squat",       weight_kg=140.0, reps=4, sets=4, rpe=8.5, velocity_ms=0.44),
            WorkoutLog(date=_days_ago(5),  exercise="bench_press", weight_kg=100.0, reps=5, sets=3, rpe=8.0, velocity_ms=0.46),
            # Week 3
            WorkoutLog(date=_days_ago(8),  exercise="bench_press", weight_kg=97.5,  reps=5, sets=4, rpe=8.0, velocity_ms=0.49),
            WorkoutLog(date=_days_ago(9),  exercise="deadlift",    weight_kg=165.0, reps=3, sets=4, rpe=8.5, velocity_ms=0.38),
            WorkoutLog(date=_days_ago(11), exercise="squat",       weight_kg=135.0, reps=5, sets=4, rpe=8.0, velocity_ms=0.47),
            WorkoutLog(date=_days_ago(12), exercise="bench_press", weight_kg=95.0,  reps=6, sets=3, rpe=7.5, velocity_ms=0.53),
            # Week 2
            WorkoutLog(date=_days_ago(15), exercise="bench_press", weight_kg=92.5,  reps=6, sets=4, rpe=7.5, velocity_ms=0.55),
            WorkoutLog(date=_days_ago(16), exercise="deadlift",    weight_kg=160.0, reps=4, sets=3, rpe=8.0, velocity_ms=0.42),
            WorkoutLog(date=_days_ago(18), exercise="squat",       weight_kg=130.0, reps=6, sets=3, rpe=7.5, velocity_ms=0.51),
            WorkoutLog(date=_days_ago(19), exercise="bench_press", weight_kg=90.0,  reps=8, sets=3, rpe=7.0, velocity_ms=0.59),
            # Week 1
            WorkoutLog(date=_days_ago(22), exercise="bench_press", weight_kg=87.5,  reps=8, sets=3, rpe=7.0, velocity_ms=0.62),
            WorkoutLog(date=_days_ago(23), exercise="deadlift",    weight_kg=152.5, reps=4, sets=4, rpe=7.5, velocity_ms=0.47),
            WorkoutLog(date=_days_ago(25), exercise="squat",       weight_kg=122.5, reps=6, sets=4, rpe=7.0, velocity_ms=0.56),
            WorkoutLog(date=_days_ago(26), exercise="bench_press", weight_kg=85.0,  reps=8, sets=4, rpe=6.5, velocity_ms=0.65),
        ]

    async def get_nutrition_logs(self, user_id: str, days: int = 7) -> list[NutritionLog]:
        # Slightly declining protein/calories over the week (mild under-eating)
        return [
            NutritionLog(date=_days_ago(1), calories=2850, protein_g=165, carbs_g=330, fats_g=85),
            NutritionLog(date=_days_ago(2), calories=2600, protein_g=150, carbs_g=300, fats_g=80),
            NutritionLog(date=_days_ago(3), calories=2400, protein_g=138, carbs_g=275, fats_g=76),
            NutritionLog(date=_days_ago(4), calories=3100, protein_g=180, carbs_g=360, fats_g=92),
            NutritionLog(date=_days_ago(5), calories=2700, protein_g=155, carbs_g=310, fats_g=84),
            NutritionLog(date=_days_ago(6), calories=2200, protein_g=120, carbs_g=255, fats_g=72),
            NutritionLog(date=_days_ago(7), calories=2950, protein_g=168, carbs_g=340, fats_g=88),
        ]

    async def get_sleep_logs(self, user_id: str, days: int = 7) -> list[SleepLog]:
        # Declining sleep trend (stress accumulation indicator)
        return [
            SleepLog(date=_days_ago(1), hours=6.5, quality_score=0.65, hrv=52.0, soreness_score=6.0),
            SleepLog(date=_days_ago(2), hours=6.0, quality_score=0.60, hrv=48.0, soreness_score=6.5),
            SleepLog(date=_days_ago(3), hours=7.5, quality_score=0.80, hrv=62.0, soreness_score=4.0),
            SleepLog(date=_days_ago(4), hours=6.5, quality_score=0.70, hrv=55.0, soreness_score=5.5),
            SleepLog(date=_days_ago(5), hours=7.0, quality_score=0.75, hrv=58.0, soreness_score=5.0),
            SleepLog(date=_days_ago(6), hours=8.0, quality_score=0.85, hrv=65.0, soreness_score=3.5),
            SleepLog(date=_days_ago(7), hours=7.5, quality_score=0.78, hrv=61.0, soreness_score=4.5),
        ]
