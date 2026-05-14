-- Find a part with Brand#23 and MED BOX
SELECT p_partkey, p_brand, p_container
FROM part
WHERE p_brand = 'Brand#23' AND p_container = 'MED BOX'
LIMIT 1
