# JDBC Step 3: The Statement

After establishing a `Connection`, you need a mechanism to send your SQL commands to the database. This is the role of the `Statement` interfaces in JDBC. They act as a vehicle for executing your SQL queries.

There are three main types of statement interfaces:
1.  **`Statement`**: For executing simple, static SQL queries.
2.  **`PreparedStatement`**: For executing pre-compiled SQL queries with parameters. **This is the preferred choice.**
3.  **`CallableStatement`**: For executing stored procedures in the database.

This note focuses on `Statement` and `PreparedStatement`.

---

## 1. `Statement` Interface

The `Statement` interface is the most basic way to execute SQL. You create a `Statement` object from your `Connection` and then use it to run a static SQL string.

**When to use it (rarely):**
-   For simple, one-off queries that have no parameters.
-   For Data Definition Language (DDL) commands like `CREATE TABLE` or `DROP TABLE`.

**Disadvantages:**
-   **Performance**: Each `Statement` is compiled by the database every time it's executed, which can be inefficient if you run the same query multiple times.
-   **Security Risk**: Concatenating user input directly into a `Statement`'s SQL string makes your application vulnerable to **SQL Injection attacks**.

### Example: Using `Statement`

```java
try (Connection conn = DriverManager.getConnection(URL, USER, PASS);
     Statement stmt = conn.createStatement()) {

    String sql = "SELECT id, name FROM products";
    ResultSet rs = stmt.executeQuery(sql);

    // Process the ResultSet...

} catch (SQLException e) {
    e.printStackTrace();
}
```

---

## 2. `PreparedStatement` Interface (Best Practice)

The `PreparedStatement` interface, which extends `Statement`, is the recommended way to execute SQL queries in almost all scenarios. It allows you to execute a pre-compiled SQL statement with placeholder parameters (`?`).

### Key Advantages of `PreparedStatement`

1.  **Prevents SQL Injection**: This is the most important benefit. By using placeholders (`?`), the `PreparedStatement` treats user-supplied data as literal values, not as executable SQL code. The driver handles the proper escaping of characters, neutralizing any malicious input.

    **SQL Injection Example (What to AVOID):**
    ```java
    // UNSAFE CODE - VULNERABLE TO SQL INJECTION
    String userId = "105 OR 1=1"; // Malicious input
    String sql = "SELECT * FROM users WHERE id = " + userId;
    Statement stmt = conn.createStatement();
    ResultSet rs = stmt.executeQuery(sql); // This would return all users!
    ```

    **Safe Code with `PreparedStatement`:**
    ```java
    // SAFE CODE
    String sql = "SELECT * FROM users WHERE id = ?";
    PreparedStatement pstmt = conn.prepareStatement(sql);
    pstmt.setString(1, "105 OR 1=1"); // The malicious string is treated as a literal value
    ResultSet rs = pstmt.executeQuery(); // This will safely find no user with that ID
    ```

2.  **Better Performance**: The SQL query is sent to the database and compiled only once. If you execute the same `PreparedStatement` multiple times with different parameters, the database can reuse the pre-compiled execution plan, which is much faster.

3.  **Improved Readability**: Using `?` placeholders makes the SQL query cleaner and easier to read compared to complex string concatenation.

### How to Use `PreparedStatement`

1.  **Create the SQL with placeholders**: Write your SQL query using `?` for any dynamic values.
    ```java
    String sql = "INSERT INTO employees (name, salary) VALUES (?, ?)";
    ```

2.  **Create the `PreparedStatement` object**:
    ```java
    PreparedStatement pstmt = connection.prepareStatement(sql);
    ```

3.  **Set the parameter values**: Use the various `setXxx()` methods (e.g., `setString()`, `setInt()`, `setDouble()`) to bind values to the placeholders. The parameters are 1-indexed.
    ```java
    pstmt.setString(1, "Jane Doe"); // Set the first '?'
    pstmt.setDouble(2, 80000.00);  // Set the second '?'
    ```

4.  **Execute the query**: Call `executeQuery()` or `executeUpdate()`.
    ```java
    int rowsAffected = pstmt.executeUpdate();
    ```

### Full Example

```java
String sql = "UPDATE employees SET salary = ? WHERE id = ?";

try (Connection conn = DriverManager.getConnection(URL, USER, PASS);
     PreparedStatement pstmt = conn.prepareStatement(sql)) {

    // Set the values for the placeholders
    pstmt.setDouble(1, 95000.00); // New salary
    pstmt.setInt(2, 101);         // Employee ID

    int rowsAffected = pstmt.executeUpdate();
    System.out.println(rowsAffected + " row(s) updated successfully.");

} catch (SQLException e) {
    e.printStackTrace();
}
```

**Conclusion:** Always favor `PreparedStatement` over `Statement` to build secure, efficient, and maintainable database applications in Java.
