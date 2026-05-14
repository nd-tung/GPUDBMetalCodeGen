-- Q17 pattern with brand/container filter, all parts
SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem l1, part
WHERE p_partkey = l1.l_partkey
  AND p_brand = 'Brand#23'
  AND p_container = 'MED BOX'
  AND l1.l_quantity < (
    SELECT 0.2 * AVG(l_quantity) FROM lineitem l2 WHERE l2.l_partkey = p_partkey
  )
