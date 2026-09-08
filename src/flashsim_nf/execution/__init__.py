"""Execution backends for validated NFs jobs."""

from .condor import CondorResources, CondorSubmission, build_condor_submission

__all__ = ["CondorResources", "CondorSubmission", "build_condor_submission"]
