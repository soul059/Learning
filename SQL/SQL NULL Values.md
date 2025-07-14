---
tags:
  - sql
  - null
  - unknown-value
  - data-integrity
---

# SQL NULL Values

In [[SQL|SQL]], [[Null Values]] is a special marker used in a column to indicate that a data value is **missing**, **unknown**, or **not applicable**. It is not the same as a zero value, an empty string (''), or a false boolean value.

## Understanding NULL

* `NULL` represents the absence of a value.
* It's a state or a marker, not a specific data type or a value itself.
* A column is allowed to contain `NULL` values unless a [[SQL Constraints|NOT NULL]] constraint is applied to it.

## Comparing with NULL

Comparisons involving `NULL` follow a three-valued logic: `TRUE`, `FALSE`, and `UNKNOWN`.

* Any comparison operator (`=`, `!=`, `%3C`, `%3E`, `<=`, `>=`) comparing a value to `NULL`, or comparing two `NULL` values, will result in **UNKNOWN**, not `TRUE` or `FALSE`.
    * `value = NULL` is UNKNOWN
    * `value != NULL` is UNKNOWN
    * `NULL = NULL` is UNKNOWN
    * `NULL != NULL` is UNKNOWN

* Because regular comparison operators don't work with `NULL` as expected, SQL provides special operators:
    * `IS NULL`: Tests if a value is `NULL`. Returns `TRUE` if the value is `NULL`, `FALSE` otherwise.
    * `IS NOT NULL`: Tests if a value is not `NULL`. Returns `TRUE` if the value is not `NULL`, `FALSE` otherwise.

**Example:**

```sql
-- Find students who have not yet declared a major (MajorID is NULL)
SELECT FirstName, LastName
FROM Students
WHERE MajorID IS NULL;
```

```sql
-- Find students who have declared a major (MajorID is NOT NULL)
SELECT FirstName, LastName
FROM Students
WHERE MajorID IS NOT NULL;
```

```sql
-- This query will NOT reliably find students with no major due to NULL comparison behavior
-- SELECT FirstName, LastName FROM Students WHERE MajorID = NULL; -- Incorrect! Will return 0 rows
```

```sql
-- This query will NOT reliably find students WITH a major
-- SELECT FirstName, LastName FROM Students WHERE MajorID != NULL; -- Incorrect! Will return 0 rows
```
In the `WHERE` clause, only rows where the condition evaluates to `TRUE` are included in the result set. Rows evaluating to `FALSE` or `UNKNOWN` are excluded.

## NULL in Aggregate Functions

[[SQL Aggregate Functions and Grouping|Aggregate functions]] (like `SUM`, `AVG`, `COUNT(column_name)`, `MIN`, `MAX`) typically **ignore** `NULL` values in the column they are operating on.

* `COUNT(*)` counts all rows, including those with `NULL` values in other columns.
* `COUNT(column_name)` counts only the non-null values in that specific column.
* `AVG()` calculates the average based only on non-null values (Sum of non-nulls / Count of non-nulls).

**Example:**

```sql
CREATE TABLE Products (
    ProductID INT PRIMARY KEY,
    Name VARCHAR(100),
    Price DECIMAL(10,2),
    Rating INT -- Could be NULL if not yet rated
);

INSERT INTO Products VALUES (1, 'Laptop', 1200.00, 4);
INSERT INTO Products VALUES (2, 'Keyboard', 75.00, NULL); -- No rating yet
INSERT INTO Products VALUES (3, 'Monitor', 300.00, 5);
INSERT INTO Products VALUES (4, 'Mouse', 25.00, 3);
INSERT INTO Products VALUES (5, 'Webcam', 50.00, NULL); -- No rating yet
```

```sql
-- Get the total number of products
SELECT COUNT(*) FROM Products; -- Result: 5
```

```sql
-- Get the number of products with a rating
SELECT COUNT(Rating) FROM Products; -- Result: 3 (Ignores NULLs)
```

```sql
-- Get the average rating
SELECT AVG(Rating) FROM Products; -- Result: (4+5+3)/3 = 4.0 (Ignores NULLs)
```

```sql
-- Get the sum of prices (NULLs in price column would be ignored)
SELECT SUM(Price) FROM Products; -- Result: 1650.00 (No NULLs in Price in this example)
```

Understanding how `NULL` behaves is crucial for writing correct and predictable SQL queries and ensuring data integrity.

---
**Related Notes:**
* [[SQL]]
* [[SQL Constraints]]
* [[SQL DQL]]
* [[SQL Aggregate Functions and Grouping]]