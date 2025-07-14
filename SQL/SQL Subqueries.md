---
tags:
  - sql
  - subquery
  - nested-query
  - dql
---

# SQL Subqueries

[[SQL|A subquery]] (also known as an inner query or nested query) is a [[SQL DQL|SELECT]] query embedded inside another SQL statement. Subqueries can be used in various clauses of the outer statement, such as `WHERE`, `FROM`, or `SELECT`.

Subqueries are executed first, and their result is then used by the outer query.

## Types of Subqueries

Subqueries are often classified by the number of columns and rows they return:

1.  **Scalar Subquery:** Returns a single value (one column, one row). Can be used anywhere a single value (an expression) is expected.
2.  **Row Subquery:** Returns a single row (one or more columns, one row). Can be used in a `WHERE` clause for row comparisons.
3.  **Table Subquery:** Returns a table (one or more columns, zero or more rows). Can be used in a `FROM` clause (treated as a derived table) or with operators that expect a set of rows (`IN`, `EXISTS`).

## Common Use Cases and Operators

### 1. In the WHERE Clause

This is the most common use. Subqueries are used to filter the outer query based on values determined by the inner query.

* **Using Comparison Operators (=, %3C, %3E, <=, >=, !=):** Used with **scalar** or **row** subqueries.

    ```sql
    -- Find students whose major is the same as StudentID 101's major
    SELECT FirstName, LastName
    FROM Students
    WHERE MajorID = (SELECT MajorID FROM Students WHERE StudentID = 101); -- Scalar subquery
    ```

* **Using `IN` / `NOT IN`:** Used with **table** subqueries that return a single column. Checks if a value exists within the set returned by the subquery.

    ```sql
    -- Find students enrolled in courses offered in the 'Fall 2024' semester
    SELECT FirstName, LastName
    FROM Students
    WHERE StudentID IN (SELECT StudentID FROM Enrollments WHERE Semester = 'Fall 2024'); -- Table subquery (single column)
    ```

* **Using `EXISTS` / `NOT EXISTS`:** Used with **table** subqueries. Checks for the *existence* of any rows returned by the subquery. It's more efficient than `IN` when you only need to check for existence.

    ```sql
    -- Find students who have enrolled in at least one course
    SELECT FirstName, LastName
    FROM Students S
    WHERE EXISTS (SELECT 1 FROM Enrollments E WHERE E.StudentID = S.StudentID); -- Correlated subquery (see below)
    ```
    *(`SELECT 1` is common practice as the specific columns don't matter, only the existence of rows).*

* **Using `ANY` / `SOME` / `ALL`:** Used with **table** subqueries that return a single column.
    * `ANY` (`SOME` is a synonym): Compares a value to *any* value in the set returned by the subquery. Needs a comparison operator (e.g., `> ANY` means greater than at least one value).
    * `ALL`: Compares a value to *all* values in the set returned by the subquery (e.g., `> ALL` means greater than every value).

    ```sql
    -- Find courses with a credit higher than ANY course in the 'Intro' category
    SELECT CourseName
    FROM Courses
    WHERE Credits > ANY (SELECT Credits FROM Courses WHERE CourseName LIKE 'Intro %');
    ```

    ```sql
    -- Find courses with a credit higher than ALL courses in the 'Advanced' category
    SELECT CourseName
    FROM Courses
    WHERE Credits > ALL (SELECT Credits FROM Courses WHERE CourseName LIKE 'Advanced %');
    ```

### 2. In the FROM Clause (Derived Tables)

A subquery in the `FROM` clause returns a result set that is treated as a temporary table, often called a **derived table** or **inline view**. This temporary table can then be queried like a regular table. Derived tables **must** be given an alias.

```sql
-- Find the average number of orders per customer
SELECT AVG(OrdersPerCustomer)
FROM (
    SELECT CustomerID, COUNT(OrderID) AS OrdersPerCustomer
    FROM Orders
    GROUP BY CustomerID
) AS CustomerOrderCounts; -- Derived table needs an alias
```

### 3. In the SELECT Clause (Scalar Subqueries)

A scalar subquery can be used as a column in the `SELECT` list. It must return at most one row and one column for each row of the outer query.

```sql
-- List each customer and the number of orders they have placed
SELECT
    CustomerID,
    (SELECT COUNT(OrderID) FROM Orders O WHERE O.CustomerID = C.CustomerID) AS NumberOfOrders -- Scalar subquery
FROM Customers C;
```
*This can often be achieved more efficiently using [[SQL Joins|LEFT JOIN]] and [[SQL Aggregate Functions and Grouping|GROUP BY]].*

## Correlated Subqueries

A correlated subquery is a subquery where the inner query depends on the outer query. It cannot be executed independently. The inner query references a column from the table in the outer query. The inner query is evaluated once for **each row** processed by the outer query.

* Often used with `EXISTS` or in the `SELECT` clause.
* Can be less efficient than non-correlated subqueries or joins, but sometimes provide the clearest way to express a logic.

**Example (revisiting the EXISTS example):**

```sql
-- Find students who have enrolled in at least one course (S.StudentID is used in the inner query)
SELECT FirstName, LastName
FROM Students S
WHERE EXISTS (SELECT 1 FROM Enrollments E WHERE E.StudentID = S.StudentID);
```

Subqueries are versatile tools that allow you to perform complex filtering and data retrieval by embedding queries within other SQL statements. However, it's often important to consider alternative approaches like [[SQL Joins|joins]] or temporary tables, as they can sometimes be more performant, especially for correlated subqueries.

---
**Related Notes:**
* [[SQL]]
* [[SQL DQL]]
* [[SQL Joins]]
* [[SQL Aggregate Functions and Grouping]]