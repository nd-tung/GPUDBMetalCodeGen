-- S13: Scalar COUNT(*)
SELECT COUNT(*) AS cnt
FROM lineitem
WHERE l_orderkey < 1000
