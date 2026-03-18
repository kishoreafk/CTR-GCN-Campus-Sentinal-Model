"""
Resolves --classes / --class_ids / --class_category / --all_classes
into a ClassRegistry subset.

Key design decisions:
- --classes uses fuzzy/partial name matching (lowercase, stripped)
  so users don't need exact AVA names like "run/jog"
- Unknown names produce a clear error listing valid options
- The resolved subset is written to a run-specific YAML for reproducibility
"""

from utils.class_registry import ClassRegistry
from typing import List
import argparse, logging, yaml
from pathlib import Path

log = logging.getLogger("class_resolver")


def resolve_classes(args: argparse.Namespace,
                    registry: ClassRegistry) -> ClassRegistry:
    """
    Entry point: inspects args, delegates to the appropriate resolver,
    returns a ClassRegistry scoped to the selected classes.

    Priority: --class_ids > --classes > --class_category > --all_classes
    If none specified: raise with helpful usage message.
    """
    if getattr(args, "class_ids", None):
        return _resolve_by_ids(args.class_ids, registry)

    if getattr(args, "classes", None):
        return _resolve_by_names(args.classes, registry)

    if getattr(args, "class_category", None):
        return _resolve_by_category(args.class_category, registry)

    if getattr(args, "all_classes", False):
        return registry   # unchanged

    # Nothing specified: error with helpful message
    raise ValueError(
        "No classes specified. Use one of:\n"
        "  --classes eat drink walk\n"
        "  --class_ids 17 20 14\n"
        "  --class_category interaction\n"
        "  --all_classes\n"
        f"\nAvailable classes:\n{_format_class_list(registry)}"
    )


def _resolve_by_ids(ids: List[int], registry: ClassRegistry) -> ClassRegistry:
    """
    Validate each ID against registry. Raise if any ID is unknown.
    Returns a new ClassRegistry scoped to the specified IDs.
    """
    valid_ids = set(registry.class_ids)
    unknown = [i for i in ids if i not in valid_ids]
    if unknown:
        raise ValueError(
            f"Unknown AVA class IDs: {unknown}\n"
            f"Valid IDs: {sorted(valid_ids)}"
        )
    selected = [c for c in registry._classes if c["id"] in ids]
    return ClassRegistry.from_class_list(selected)


def _resolve_by_names(names: List[str],
                      registry: ClassRegistry) -> ClassRegistry:
    """
    Partial case-insensitive name match.
    'run' matches 'run/jog', 'punch' matches 'punch/slap', etc.
    Raises with suggestions if no match.
    """
    selected = []
    not_found = []
    for name in names:
        query = name.lower().strip()
        matches = [c for c in registry._classes
                   if query in c["name"].lower()]
        if not matches:
            not_found.append(name)
        else:
            # If multiple matches, pick most specific (shortest name)
            best = min(matches, key=lambda c: len(c["name"]))
            if best not in selected:
                selected.append(best)
            if len(matches) > 1:
                log.warning(
                    f"'{name}' matched multiple classes: "
                    f"{[c['name'] for c in matches]}. "
                    f"Using '{best['name']}'. "
                    f"Use --class_ids {best['id']} to be unambiguous."
                )

    if not_found:
        raise ValueError(
            f"No matching classes for: {not_found}\n"
            f"Available:\n{_format_class_list(registry)}"
        )

    return ClassRegistry.from_class_list(selected)


def _resolve_by_category(category: str,
                          registry: ClassRegistry) -> ClassRegistry:
    """Select all classes in a given category."""
    selected = [c for c in registry._classes
                if c["category"] == category]
    if not selected:
        cats = sorted(set(c["category"] for c in registry._classes))
        raise ValueError(
            f"No classes in category '{category}'. "
            f"Available categories: {cats}"
        )
    log.info(f"Category '{category}': "
             f"{[c['name'] for c in selected]}")
    return ClassRegistry.from_class_list(selected)


def _format_class_list(registry: ClassRegistry) -> str:
    lines = []
    by_cat = {}
    for c in registry._classes:
        by_cat.setdefault(c["category"], []).append(c)
    for cat, classes in sorted(by_cat.items()):
        lines.append(f"\n  [{cat}]")
        for c in classes:
            lines.append(f"    id={c['id']:3d}  {c['name']}")
    return "\n".join(lines)


def save_class_selection(selected: ClassRegistry, run_dir: str):
    """
    Persist the resolved class list to {run_dir}/selected_classes.yaml.
    Loaded by evaluation/reporting to know which classes were trained on.
    """
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{run_dir}/selected_classes.yaml", "w") as f:
        yaml.dump({
            "num_classes": selected.num_classes,
            "classes":     selected._classes,
        }, f)
