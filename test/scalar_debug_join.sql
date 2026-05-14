-- Find a matching partkey
SELECT p_partkey 
FROM part
WHERE p_brand = 'Brand#23' AND p_container = 'MED BOX'
LIMIT 3
