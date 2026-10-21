"""Forensic image preprocessing modules (Error Level Analysis & Laplacian micro-edges)"""
from .ela import compute_ela
from .edges import compute_laplacian_edges

__all__ = ["compute_ela", "compute_laplacian_edges"]
