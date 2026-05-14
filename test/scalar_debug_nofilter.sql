-- Q17 without the scalar subquery filter
SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem l1, part
WHERE p_partkey = l1.l_partkey
  AND p_brand = 'Brand#23'
  AND p_container = 'MED BOX'
