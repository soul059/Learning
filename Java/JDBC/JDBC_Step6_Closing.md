# JDBC Step 6: Closing Resources

One of the most critical aspects of working with JDBC is proper resource management. JDBC objects like `Connection`, `Statement`, and `ResultSet` are expensive resources that consume both memory in your Java application and resources on the database server (like cursors and network sockets).

Failing to close these resources can lead to serious problems, including:
-   **Resource Leaks**: The database server may run out of cursors or connections.
-   **Memory Leaks**: Your Java application's memory usage may grow indefinitely.
-   **Application Instability**: The application may become slow and eventually crash.

---

## The Importance of the `finally` Block (The Old Way)

Before Java 7, the standard way to ensure resources were closed was to use a `finally` block. The code inside a `finally` block is **guaranteed to execute**, regardless of whether the `try` block completes normally or throws an exception.

This pattern involved a lot of boilerplate and nested `try-catch` blocks, making the code verbose and hard to read.

### Example: The Verbose `finally` Block Pattern

```java
Connection connection = null;
Statement statement = null;
ResultSet resultSet = null;

try {
    // 1. Get Connection
    connection = DriverManager.getConnection(URL, USER, PASS);
    // 2. Create Statement
    statement = connection.createStatement();
    // 3. Execute Query
    resultSet = statement.executeQuery("SELECT * FROM users");
    // 4. Process ResultSet
    while (resultSet.next()) {
        // ...
    }
} catch (SQLException e) {
    e.printStackTrace();
} finally {
    // 5. Close resources in reverse order of creation
    if (resultSet != null) {
        try {
            resultSet.close();
        } catch (SQLException e) { /* ignored */ }
    }
    if (statement != null) {
        try {
            statement.close();
        } catch (SQLException e) { /* ignored */ }
    }
    if (connection != null) {
        try {
            connection.close();
        } catch (SQLException e) { /* ignored */ }
    }
}
```
This code is cumbersome. Notice the nested `try-catch` blocks within the `finally` block, which are necessary because the `close()` methods themselves can throw a `SQLException`.

---

## The `try-with-resources` Statement (The Modern, Best Practice Way)

Java 7 introduced the `try-with-resources` statement, which dramatically simplifies resource management. It automatically closes any resource that implements the `java.lang.AutoCloseable` interface (which `Connection`, `Statement`, and `ResultSet` all do).

**This is the recommended and standard way to manage JDBC resources.**

### How It Works

You declare and initialize your resources within the parentheses `()` following the `try` keyword. The Java compiler automatically generates the necessary `finally` block to close these resources in the correct order (the reverse of their declaration).

### Syntax

```java
try (
    // Declare and initialize resources here
    ResourceType resource1 = new ResourceType();
    ResourceType resource2 = new ResourceType()
) {
    // Use the resources here
    // ...
} catch (Exception e) {
    // Handle exceptions
    // ...
}
// The resources are automatically closed here, whether an
// exception was thrown or not.
```

### Example: JDBC with `try-with-resources`

Here is the same logic as the previous example, but rewritten using `try-with-resources`. The code is far more concise, readable, and less error-prone.

```java
String sql = "SELECT id, name FROM users";

// Resources declared in the try() block will be automatically closed.
try (Connection conn = DriverManager.getConnection(URL, USER, PASS);
     Statement stmt = conn.createStatement();
     ResultSet rs = stmt.executeQuery(sql)) {

    // Process the ResultSet
    while (rs.next()) {
        System.out.printf("ID: %d, Name: %s%n", rs.getInt("id"), rs.getString("name"));
    }

} catch (SQLException e) {
    // A single catch block is sufficient.
    e.printStackTrace();
}
// No 'finally' block is needed! conn, stmt, and rs are all closed.
```

### Why `try-with-resources` is Superior

1.  **Readability**: The code is much cleaner and focuses on the core logic, not on the boilerplate of resource cleanup.
2.  **Safety**: It's impossible to forget to close a resource. The compiler handles it for you.
3.  **Correctness**: It correctly handles exceptions that might occur during the closing of resources themselves (suppressed exceptions), a detail that is difficult to get right with manual `finally` blocks.

**Conclusion:** Always use the `try-with-resources` statement for managing your JDBC resources. It is the modern standard for writing safe and clean JDBC code.
