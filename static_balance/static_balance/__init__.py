# SPDX-License-Identifier: GPL-3.0-or-later

"""Static balance of robots making contacts."""

from .contact import Contact, FlatContact, PointContact
from .contact_set import ContactSet
from .force_distribution import compute_supporting_forces
from .static_equilibrium import compute_static_equilibrium_polygon

__version__ = "1.0.0"

__all__ = [
    "Contact",
    "ContactSet",
    "FlatContact",
    "PointContact",
    "compute_static_equilibrium_polygon",
    "compute_supporting_forces",
]
