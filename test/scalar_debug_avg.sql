-- Cross-table correlated: same pattern as Q17 but with all parts
SELECT l_orderkey, l_linenumber, l_quantity, l_partkey
FROM lineitem l1, part
WHERE p_partkey = l1.l_partkey
  AND l1.l_partkey BETWEEN 1 AND 3
  AND l1.l_quantity < (
    SELECT 0.2 * AVG(l_quantity) FROM lineitem l2 WHERE l2.l_partkey = p_partkey
  )
ORDER BY l1.l_partkey, l1.l_quantity
LIMIT 20
