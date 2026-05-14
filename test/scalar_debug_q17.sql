-- Debug Q17: check avg per partkey for single part
SELECT l_partkey, COUNT(*) as cnt, SUM(l_quantity) as sum_qty, SUM(l_quantity)/COUNT(*) as avg_qty
FROM lineitem WHERE l_partkey BETWEEN 1 AND 3
GROUP BY l_partkey
