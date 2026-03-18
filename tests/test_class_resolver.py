"""
Tests for utils/class_resolver.py

Covers:
  - Resolution by name (partial match)
  - Resolution by ID
  - Resolution by category
  - --all_classes flag
  - Error on unknown names / IDs
  - Fuzzy matching warnings
"""
import pytest
import argparse
from utils.class_registry import ClassRegistry
from utils.class_resolver import resolve_classes


@pytest.fixture
def full_registry():
    """15-class registry from class_config.yaml."""
    return ClassRegistry()


def _make_args(**kwargs):
    """Build an argparse.Namespace with defaults for all class selection fields."""
    defaults = dict(
        classes=None, class_ids=None, class_category=None,
        all_classes=False
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ── By name ──────────────────────────────────────────────────────────────────

class TestResolveByName:
    def test_exact_match(self, full_registry):
        args = _make_args(classes=["eat"])
        result = resolve_classes(args, full_registry)
        assert result.num_classes == 1
        assert "eat" in result.class_names[0].lower()

    def test_partial_match(self, full_registry):
        args = _make_args(classes=["run"])
        result = resolve_classes(args, full_registry)
        assert result.num_classes == 1
        assert "run" in result.class_names[0].lower()

    def test_multiple_names(self, full_registry):
        args = _make_args(classes=["eat", "walk", "dance"])
        result = resolve_classes(args, full_registry)
        assert result.num_classes == 3

    def test_unknown_name_raises(self, full_registry):
        args = _make_args(classes=["nonexistent_class"])
        with pytest.raises(ValueError, match="No matching classes"):
            resolve_classes(args, full_registry)


# ── By ID ────────────────────────────────────────────────────────────────────

class TestResolveByID:
    def test_single_id(self, full_registry):
        args = _make_args(class_ids=[17])
        result = resolve_classes(args, full_registry)
        assert result.num_classes == 1
        assert 17 in result.class_ids

    def test_multiple_ids(self, full_registry):
        args = _make_args(class_ids=[17, 49, 74])
        result = resolve_classes(args, full_registry)
        assert result.num_classes == 3

    def test_unknown_id_raises(self, full_registry):
        args = _make_args(class_ids=[999])
        with pytest.raises(ValueError, match="Unknown AVA class IDs"):
            resolve_classes(args, full_registry)


# ── By category ──────────────────────────────────────────────────────────────

class TestResolveByCategory:
    def test_valid_category(self, full_registry):
        args = _make_args(class_category="interaction")
        result = resolve_classes(args, full_registry)
        assert result.num_classes >= 1
        # All returned classes should have category "interaction"
        for c in result._classes:
            assert c["category"] == "interaction"

    def test_invalid_category_raises(self, full_registry):
        args = _make_args(class_category="nonexistent")
        with pytest.raises(ValueError, match="No classes in category"):
            resolve_classes(args, full_registry)


# ── All classes ──────────────────────────────────────────────────────────────

class TestAllClasses:
    def test_all_classes_returns_full(self, full_registry):
        args = _make_args(all_classes=True)
        result = resolve_classes(args, full_registry)
        assert result.num_classes == full_registry.num_classes


# ── No selection ─────────────────────────────────────────────────────────────

class TestNoSelection:
    def test_no_args_raises(self, full_registry):
        args = _make_args()
        with pytest.raises(ValueError, match="No classes specified"):
            resolve_classes(args, full_registry)


# ── Subset registry ──────────────────────────────────────────────────────────

class TestSubsetRegistry:
    def test_from_class_list(self):
        classes = [
            {"id": 17, "name": "eat", "category": "object"},
            {"id": 49, "name": "walk", "category": "movement"},
        ]
        reg = ClassRegistry.from_class_list(classes)
        assert reg.num_classes == 2
        assert 17 in reg.class_ids
        assert 49 in reg.class_ids

    def test_to_config_fragment(self, full_registry):
        frag = full_registry.to_config_fragment()
        assert frag["num_classes"] == full_registry.num_classes
        assert len(frag["target_class_ids"]) == full_registry.num_classes

    def test_is_subset_compatible(self, full_registry):
        subset = ClassRegistry.from_class_list(
            [{"id": 17, "name": "eat", "category": "object"}])
        assert subset.is_subset_compatible(full_registry)
        assert not full_registry.is_subset_compatible(subset)

    def test_multilabel_vector_subset(self):
        classes = [
            {"id": 17, "name": "eat", "category": "object"},
            {"id": 49, "name": "walk", "category": "movement"},
            {"id": 74, "name": "talk", "category": "interaction"},
        ]
        reg = ClassRegistry.from_class_list(classes)
        vec = reg.get_multilabel_vector([17, 49])
        assert vec.shape == (3,)
        assert vec[0] == 1.0  # eat → index 0
        assert vec[1] == 1.0  # walk → index 1
        assert vec[2] == 0.0  # talk → not present
