use minitorch_rs::operators::*;
use proptest::prelude::*;

/// Helper: assert two f64s are close (within 1e-2), with a proptest-friendly message.
fn assert_close(a: f64, b: f64) {
    assert!(
        is_close(a, b),
        "assert_close failed: {a} vs {b} (diff = {})",
        (a - b).abs()
    );
}

/// Strategy matching minitorch's `small_floats` — avoids extreme values
/// that blow up exp/log but covers a wide-enough range.
fn small_floats() -> impl Strategy<Value = f64> {
    -100.0f64..100.0
}

// ===================================================================
// Task 0.1 — Basic hypothesis tests
// ===================================================================

proptest! {
    #[test]
    fn test_same_as_python(x in small_floats(), y in small_floats()) {
        assert_close(mul(x, y), x * y);
        assert_close(add(x, y), x + y);
        assert_close(neg(x), -x);
        assert_close(max(x, y), if x > y { x } else { y });
        if x.abs() > 1e-5 {
            assert_close(inv(x), 1.0 / x);
        }
    }

    #[test]
    fn test_relu(a in small_floats()) {
        if a > 0.0 {
            prop_assert!((relu(a) - a).abs() < f64::EPSILON);
        }
        if a < 0.0 {
            prop_assert!((relu(a) - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_relu_back(a in small_floats(), b in small_floats()) {
        if a > 0.0 {
            prop_assert!((relu_back(a, b) - b).abs() < f64::EPSILON);
        }
        if a < 0.0 {
            prop_assert!((relu_back(a, b) - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_id(a in small_floats()) {
        prop_assert!((id(a) - a).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lt(a in small_floats()) {
        assert_close(lt(a - 1.0, a), 1.0);
        assert_close(lt(a, a - 1.0), 0.0);
    }

    #[test]
    fn test_max(a in small_floats()) {
        assert_close(max(a - 1.0, a), a);
        assert_close(max(a, a - 1.0), a);
        assert_close(max(a + 1.0, a), a + 1.0);
        assert_close(max(a, a + 1.0), a + 1.0);
    }

    #[test]
    fn test_eq(a in small_floats()) {
        assert_close(eq(a, a), 1.0);
        assert_close(eq(a, a - 1.0), 0.0);
        assert_close(eq(a, a + 1.0), 0.0);
    }
}

// ===================================================================
// Task 0.2 — Property testing (YOU implement these)
// ===================================================================

proptest! {
    /// Check properties of sigmoid:
    /// * Always between 0.0 and 1.0
    /// * 1 - sigmoid(x) ≈ sigmoid(-x)
    /// * sigmoid(0) = 0.5
    /// * Strictly increasing
    #[test]
    fn test_sigmoid(a in small_floats()) {
        let s = sigmoid(a);
        // Always in [0, 1]
        prop_assert!(s >= 0.0 && s <= 1.0);
        // Symmetry: sigmoid(a) + sigmoid(-a) ≈ 1
        assert_close(s + sigmoid(-a), 1.0);
        prop_assert!(sigmoid(0.) == 0.5);
        prop_assert!(sigmoid(a) <= sigmoid(a + 0.01));
    }

    /// Test transitive property: a < b and b < c implies a < c
    #[test]
    fn test_transitive(a in small_floats(), b in small_floats(), c in small_floats()) {
        if lt(a, b) == 1.0 && lt(b, c) == 1.0 { prop_assert!(lt(a, c) == 1.); }
    }

    /// Test that mul is symmetric: mul(x, y) == mul(y, x)
    #[test]
    fn test_symmetric(x in small_floats(), y in small_floats()) {
        prop_assert!(mul(x, y) == mul(y, x));
    }

    /// Test distributive property: z * (x + y) = z * x + z * y
    #[test]
    fn test_distribute(x in small_floats(), y in small_floats(), z in small_floats()) {
        prop_assert!(is_close(mul(z, add(x, y)), add(mul(z, x), mul(z, y))));
    }

    /// Test some other property of your choosing.
    #[test]
    fn test_other(x in small_floats()) {
        prop_assert!(is_close(x, add(x, 0.)));
        prop_assert!(is_close(x, mul(x, 1.)));
    }
}

// ===================================================================
// Task 0.3 — Higher-order functions
// These require neg_list, add_lists, sum, prod to be implemented.
// Uncomment once you've written them in operators.rs.
// ===================================================================

proptest! {
    #[test]
    fn test_zip_with(a in small_floats(), b in small_floats(),
                     c in small_floats(), d in small_floats()) {
        let result = add_lists(&[a, b], &[c, d]);
        assert_close(result[0], a + c);
        assert_close(result[1], b + d);
    }

    #[test]
    fn test_sum_distribute(
        ls1 in proptest::collection::vec(small_floats(), 5),
        ls2 in proptest::collection::vec(small_floats(), 5),
    ) {
        assert_close(sum(&add_lists(&ls1, &ls2)), add(sum(&ls1), sum(&ls2)));
    }

    #[test]
    fn test_sum(ls in proptest::collection::vec(small_floats(), 0..20)) {
        let s = sum(&ls);
        let expected: f64 = ls.iter().sum();
        assert_close(s, expected);
    }

    #[test]
    fn test_prod(x in small_floats(), y in small_floats(), z in small_floats()) {
        assert_close(prod(&[x, y, z]), x * y * z);
    }

    #[test]
    fn test_neg_list(ls in proptest::collection::vec(small_floats(), 0..20)) {
        let result = neg_list(&ls);
        for (i, j) in ls.iter().zip(result.iter()) {
            assert_close(*i, -j);
        }
    }
}

// ===================================================================
// Back functions — just check they don't panic
// ===================================================================

proptest! {
    #[test]
    fn test_backs(a in small_floats(), b in small_floats()) {
        relu_back(a, b);
        inv_back(a + 2.4, b);
        log_back(a.abs() + 4.0, b);
    }
}
