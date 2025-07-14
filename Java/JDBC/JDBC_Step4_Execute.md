# JDBC Step 4: Executing the Query

Once you have created a `Statement` or `PreparedStatement` object, the next step is to execute it. The method you use depends on the type of SQL statement you are running (i.e., whether you are retrieving data or modifying it).

The `Statement` and `PreparedStatement` interfaces provide three main execution methods.

---

## 1. `executeQuery()`

This method is used for SQL statements that are expected to return data, which are almost always `SELECT` statements.

-   **Return Value**: It returns a `java.sql.ResultSet` object. The `ResultSet` contains all the rows of data that match the query's criteria.
-   **Usage**: Use this method when you need to retrieve information from the database.

### Example with `PreparedStatement`

```java
String sql = "SELECT id, name, email FROM customers WHERE city = ?";
ResultSet resultSet = null;

try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
    
    pstmt.setString(1, "New York"); // Set the parameter for the query
    
    // Execute the SELECT query
    resultSet = pstmt.executeQuery();
    
    // Now, process the resultSet to get the data
    while (resultSet.next()) {
        // ... process each row
    }
    
} catch (SQLException e) {
    e.printStackTrace();
} finally {
    // It's good practice to close the ResultSet in a finally block
    // if not using try-with-resources for it.
    if (resultSet != null) {
        try {
            resultSet.close();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
```
**Note:** A `ResultSet` should also be closed when you are finished with it. A `try-with-resources` block is the best way to manage this.

---

## 2. `executeUpdate()`

This method is used for SQL statements that modify data in the database. This includes `INSERT`, `UPDATE`, and `DELETE` statements. It can also be used for Data Definition Language (DDL) statements like `CREATE TABLE`, `DROP TABLE`, and `ALTER TABLE`.

-   **Return Value**: It returns an `int`.
    -   For `INSERT`, `UPDATE`, or `DELETE`, the `int` represents the **number of rows affected** by the operation.
    -   For DDL statements, the return value is typically `0`.
-   **Usage**: Use this method when you are changing the state or structure of the database.

### Example with `PreparedStatement`

```java
String sql = "UPDATE products SET price = ? WHERE id = ?";

try (PreparedStatement pstmt = connection.prepareStatement(sql)) {
    
    pstmt.setDouble(1, 19.99); // Set the new price
    pstmt.setInt(2, 101);      // Set the product ID
    
    // Execute the UPDATE statement
    int rowsAffected = pstmt.executeUpdate();
    
    System.out.println(rowsAffected + " row(s) were updated successfully.");
    
    if (rowsAffected > 0) {
        // The update was successful
    } else {
        // The update did not affect any rows (e.g., product with ID 101 not found)
    }
    
} catch (SQLException e) {
    e.printStackTrace();
}
```

---

## 3. `execute()`

This method is more general and can handle any type of SQL statement. It's useful when you are working with a dynamic SQL statement and don't know in advance whether it will be a `SELECT` or an update.

-   **Return Value**: It returns a `boolean`.
    -   `true`: The result is a `ResultSet` (i.e., it was a `SELECT` statement).
    -   `false`: The result is an update count or there is no result (i.e., it was an `INSERT`, `UPDATE`, `DELETE`, or DDL statement).

-   **Usage**: Typically used in frameworks or tools that need to execute arbitrary SQL, but less common in day-to-day application code.

### How to Use `execute()`

If `execute()` returns `true`, you can get the `ResultSet` by calling `getResultSet()`. If it returns `false`, you can get the number of affected rows by calling `getUpdateCount()`.

### Example

```java
String dynamicSql = getDynamicSql(); // Method that returns some SQL string

try (Statement stmt = connection.createStatement()) {
    
    boolean isResultSet = stmt.execute(dynamicSql);
    
    if (isResultSet) {
        // It was a SELECT statement
        try (ResultSet rs = stmt.getResultSet()) {
            System.out.println("Query returned a result set:");
            while (rs.next()) {
                // ... process results
            }
        }
    } else {
        // It was an update or DDL statement
        int updateCount = stmt.getUpdateCount();
        if (updateCount != -1) {
            System.out.println("Query was an update. " + updateCount + " row(s) affected.");
        } else {
            System.out.println("Query did not return a result set or an update count.");
        }
    }
    
} catch (SQLException e) {
    e.printStackTrace();
}
```

## Summary

| Method            | Used For                               | Returns                               |
|-------------------|----------------------------------------|---------------------------------------|
| `executeQuery()`  | `SELECT` statements                    | `ResultSet`                           |
| `executeUpdate()` | `INSERT`, `UPDATE`, `DELETE`, DDL      | `int` (number of rows affected)       |
| `execute()`       | Any type of SQL statement              | `boolean` (true if `ResultSet`)       |

For most application development, you will primarily use `executeQuery()` for reading data and `executeUpdate()` for modifying data.
