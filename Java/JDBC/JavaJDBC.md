# A Guide to JDBC with PostgreSQL

JDBC (Java Database Connectivity) is a standard Java API for connecting Java applications to relational databases. It provides a common interface to interact with various database systems, allowing developers to write database-agnostic code.

This guide provides a detailed walkthrough of using JDBC to connect to and interact with a **PostgreSQL** database.

---

## 1. What is JDBC?

JDBC is a set of Java classes and interfaces that allow a Java application to send SQL statements to a database and process the results. The API is part of the standard Java SE platform.

The key components of the JDBC architecture are:
-   **JDBC API**: Provides the interfaces (`Connection`, `Statement`, `ResultSet`, etc.) that your application uses.
-   **Driver Manager**: Manages the list of database drivers. It uses the connection URL you provide to find and load the appropriate driver.
-   **JDBC Driver**: A specific implementation of the JDBC API that understands how to communicate with a particular database (e.g., the PostgreSQL JDBC driver).
-   **Database**: The actual database system, like PostgreSQL.

---

## 2. Seven Steps to Connect and Query

Connecting to a database with JDBC follows a standard sequence of steps.

### Step 1: Add the JDBC Driver Dependency

Before you can connect to PostgreSQL, you need to add its JDBC driver to your project's classpath. If you are using a build tool like Maven or Gradle, this is the recommended approach.

**Maven (`pom.xml`):**
```xml
<dependency>
    <groupId>org.postgresql</groupId>
    <artifactId>postgresql</artifactId>
    <version>42.7.3</version> <!-- Check for the latest version -->
</dependency>
```

**Gradle (`build.gradle`):**
```groovy
implementation 'org.postgresql:postgresql:42.7.3' // Check for the latest version
```
If you are not using a build tool, you must download the PostgreSQL JDBC driver JAR file and add it to your project's build path manually.

### Step 2: Load and Register the Driver (Often Automatic)

In modern JDBC (version 4.0 and later), the driver is loaded and registered automatically as long as the driver JAR is on the classpath. The `DriverManager` finds the driver using the Service Provider Interface (SPI) mechanism.

You generally **do not** need to write this line anymore:
`Class.forName("org.postgresql.Driver");`

### Step 3: Establish the Connection

You create a `Connection` object by calling `DriverManager.getConnection()`. You need to provide three key pieces of information:
1.  **Database URL**: A string with a specific format for the database.
2.  **Username**: The database user.
3.  **Password**: The user's password.

The JDBC URL format for PostgreSQL is:
`jdbc:postgresql://<host>:<port>/<databaseName>`

**Example:**
```java
String url = "jdbc:postgresql://localhost:5432/mydatabase";
String user = "myuser";
String password = "mypassword";

try (Connection connection = DriverManager.getConnection(url, user, password)) {
    // Connection established! Use the connection object here.
    System.out.println("Connected to the PostgreSQL server successfully.");
} catch (SQLException e) {
    System.out.println(e.getMessage());
}
```
Using a `try-with-resources` block is highly recommended as it automatically closes the connection.

### Step 4: Create a Statement

Once you have a connection, you need a `Statement` object to execute SQL queries. The best practice is to use `PreparedStatement`.

-   **`Statement`**: Used for simple, static SQL queries.
-   **`PreparedStatement`**: Used for dynamic queries with parameters. It is pre-compiled for better performance and helps prevent **SQL injection attacks**. **Always prefer `PreparedStatement`**.

```java
String sql = "SELECT id, name, email FROM users WHERE id = ?";
PreparedStatement preparedStatement = connection.prepareStatement(sql);
```

### Step 5: Execute the Query

You use the statement object to execute your SQL.
-   **`executeQuery()`**: Used for `SELECT` statements that return data. It returns a `ResultSet` object.
-   **`executeUpdate()`**: Used for `INSERT`, `UPDATE`, or `DELETE` statements. It returns an `int` representing the number of rows affected.

```java
// For a SELECT query
preparedStatement.setInt(1, 101); // Set the first '?' parameter to 101
ResultSet resultSet = preparedStatement.executeQuery();

// For an INSERT/UPDATE/DELETE query
int rowsAffected = preparedStatement.executeUpdate();
```

### Step 6: Process the Result Set

If you executed a `SELECT` query, the results are stored in a `ResultSet` object. You can iterate through it using a `while` loop.

```java
while (resultSet.next()) {
    // Retrieve data by column name or index (1-based)
    int id = resultSet.getInt("id");
    String name = resultSet.getString("name");
    String email = resultSet.getString("email");

    System.out.println("ID: " + id + ", Name: " + name + ", Email: " + email);
}
```

### Step 7: Close the Resources

It is crucial to close all JDBC resources (`Connection`, `Statement`, `ResultSet`) to release database and memory resources. The `try-with-resources` statement is the best way to ensure this happens automatically, even if exceptions occur.

---

## 4. Complete Code Examples

Let's assume we have a table named `employees`:
```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    position VARCHAR(100),
    salary NUMERIC(10, 2)
);
```

### Example 1: SELECT Query

```java
import java.sql.*;

public class SelectExample {
    public static void main(String[] args) {
        String url = "jdbc:postgresql://localhost:5432/companydb";
        String user = "admin";
        String password = "password";

        String sql = "SELECT id, name, position, salary FROM employees WHERE salary > ?";

        try (Connection conn = DriverManager.getConnection(url, user, password);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {

            pstmt.setDouble(1, 50000.00); // Find employees with salary > 50000

            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    System.out.printf("ID: %d, Name: %s, Position: %s, Salary: %.2f%n",
                            rs.getInt("id"),
                            rs.getString("name"),
                            rs.getString("position"),
                            rs.getDouble("salary"));
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
```

### Example 2: INSERT, UPDATE, and DELETE

```java
import java.sql.*;

public class ModifyDataExample {
    private static final String URL = "jdbc:postgresql://localhost:5432/companydb";
    private static final String USER = "admin";
    private static final String PASSWORD = "password";

    public static void main(String[] args) {
        insertEmployee("John Doe", "Software Engineer", 90000.00);
        updateEmployeeSalary(1, 95000.00);
        deleteEmployee(2);
    }

    public static void insertEmployee(String name, String position, double salary) {
        String sql = "INSERT INTO employees(name, position, salary) VALUES(?, ?, ?)";
        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, name);
            pstmt.setString(2, position);
            pstmt.setDouble(3, salary);
            int rowsAffected = pstmt.executeUpdate();
            System.out.println(rowsAffected + " row(s) inserted.");
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    // ... similar methods for update and delete
}
```

---

## 5. Transaction Management

By default, JDBC operates in **auto-commit mode**, meaning each SQL statement is treated as a separate transaction and is committed immediately. For operations that require multiple statements to succeed or fail as a single unit, you must manage transactions manually.

-   **`connection.setAutoCommit(false);`**: Disables auto-commit mode.
-   **`connection.commit();`**: Commits all changes since the last commit.
-   **`connection.rollback();`**: Discards all changes since the last commit.

```java
Connection conn = null;
try {
    conn = DriverManager.getConnection(URL, USER, PASSWORD);
    conn.setAutoCommit(false); // Start transaction

    // 1. Debit from one account
    // 2. Credit to another account

    conn.commit(); // Commit transaction if both operations succeed
    System.out.println("Transaction successful.");

} catch (SQLException e) {
    if (conn != null) {
        try {
            conn.rollback(); // Roll back transaction on error
            System.err.println("Transaction rolled back.");
        } catch (SQLException ex) {
            ex.printStackTrace();
        }
    }
    e.printStackTrace();
} finally {
    if (conn != null) {
        try {
            conn.setAutoCommit(true); // Reset auto-commit
            conn.close();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
```
This ensures data integrity. If the credit operation fails, the debit operation is rolled back.
