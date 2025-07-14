# Exception Handling in Java

## 1. Exception Hierarchy

### Java Exception Class Hierarchy
```
java.lang.Object
    └── java.lang.Throwable
        ├── java.lang.Error
        │   ├── OutOfMemoryError
        │   ├── StackOverflowError
        │   └── VirtualMachineError
        └── java.lang.Exception
            ├── java.lang.RuntimeException (Unchecked)
            │   ├── NullPointerException
            │   ├── IllegalArgumentException
            │   ├── IndexOutOfBoundsException
            │   └── ClassCastException
            └── Checked Exceptions
                ├── IOException
                ├── SQLException
                ├── ClassNotFoundException
                └── InterruptedException
```

### Types of Exceptions
```java
public class ExceptionTypes {
    
    // Checked Exception - must be handled or declared
    public void readFile(String filename) throws IOException {
        FileReader file = new FileReader(filename);
        // IOException is checked - must be handled
    }
    
    // Unchecked Exception - optional to handle
    public void divideNumbers(int a, int b) {
        int result = a / b; // ArithmeticException is unchecked
    }
    
    // Error - usually not handled by application code
    public void causeStackOverflow() {
        causeStackOverflow(); // StackOverflowError
    }
    
    public static void main(String[] args) {
        ExceptionTypes example = new ExceptionTypes();
        
        // Must handle checked exception
        try {
            example.readFile("nonexistent.txt");
        } catch (IOException e) {
            System.out.println("File not found: " + e.getMessage());
        }
        
        // Optional to handle unchecked exception
        try {
            example.divideNumbers(10, 0);
        } catch (ArithmeticException e) {
            System.out.println("Division by zero: " + e.getMessage());
        }
    }
}
```

## 2. Basic Exception Handling

### try-catch-finally Blocks
```java
import java.io.*;
import java.util.Scanner;

public class BasicExceptionHandling {
    
    public void basicTryCatch() {
        try {
            int[] numbers = {1, 2, 3};
            System.out.println(numbers[5]); // IndexOutOfBoundsException
        } catch (IndexOutOfBoundsException e) {
            System.out.println("Array index out of bounds: " + e.getMessage());
        }
    }
    
    public void multipleCatchBlocks() {
        try {
            String str = null;
            int length = str.length(); // NullPointerException
            int result = 10 / 0;       // ArithmeticException
        } catch (NullPointerException e) {
            System.out.println("Null pointer exception: " + e.getMessage());
        } catch (ArithmeticException e) {
            System.out.println("Arithmetic exception: " + e.getMessage());
        } catch (Exception e) {
            System.out.println("General exception: " + e.getMessage());
        }
    }
    
    public void tryWithFinally() {
        FileInputStream fis = null;
        try {
            fis = new FileInputStream("test.txt");
            // Read file operations
        } catch (FileNotFoundException e) {
            System.out.println("File not found: " + e.getMessage());
        } finally {
            // This block always executes
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException e) {
                    System.out.println("Error closing file: " + e.getMessage());
                }
            }
            System.out.println("Finally block executed");
        }
    }
    
    public void multiCatchBlock() {
        try {
            // Some operation that might throw different exceptions
            int value = Integer.parseInt("abc");
        } catch (NumberFormatException | NullPointerException e) {
            // Handle multiple exception types in one catch block (Java 7+)
            System.out.println("Number format or null pointer exception: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        BasicExceptionHandling example = new BasicExceptionHandling();
        
        example.basicTryCatch();
        example.multipleCatchBlocks();
        example.tryWithFinally();
        example.multiCatchBlock();
    }
}
```

### Exception Information
```java
public class ExceptionInformation {
    
    public void demonstrateExceptionMethods() {
        try {
            throwException();
        } catch (Exception e) {
            System.out.println("Exception class: " + e.getClass().getName());
            System.out.println("Exception message: " + e.getMessage());
            System.out.println("String representation: " + e.toString());
            
            // Print stack trace
            System.out.println("\nStack trace:");
            e.printStackTrace();
            
            // Get stack trace elements
            StackTraceElement[] stackTrace = e.getStackTrace();
            for (StackTraceElement element : stackTrace) {
                System.out.println("Method: " + element.getMethodName() + 
                                 " in class: " + element.getClassName() + 
                                 " at line: " + element.getLineNumber());
            }
            
            // Cause chain
            Throwable cause = e.getCause();
            if (cause != null) {
                System.out.println("Caused by: " + cause.getMessage());
            }
        }
    }
    
    private void throwException() throws Exception {
        try {
            int result = 10 / 0;
        } catch (ArithmeticException e) {
            throw new Exception("Custom exception message", e);
        }
    }
    
    public static void main(String[] args) {
        new ExceptionInformation().demonstrateExceptionMethods();
    }
}
```

## 3. try-with-resources Statement

### Automatic Resource Management
```java
import java.io.*;
import java.util.Scanner;

public class TryWithResources {
    
    // Before Java 7 - manual resource management
    public void oldWayResourceManagement() {
        FileInputStream fis = null;
        BufferedReader reader = null;
        
        try {
            fis = new FileInputStream("test.txt");
            reader = new BufferedReader(new InputStreamReader(fis));
            
            String line = reader.readLine();
            System.out.println(line);
            
        } catch (IOException e) {
            System.out.println("IO Error: " + e.getMessage());
        } finally {
            // Manual cleanup - error-prone
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    System.out.println("Error closing reader: " + e.getMessage());
                }
            }
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException e) {
                    System.out.println("Error closing stream: " + e.getMessage());
                }
            }
        }
    }
    
    // Java 7+ - try-with-resources
    public void newWayResourceManagement() {
        try (FileInputStream fis = new FileInputStream("test.txt");
             BufferedReader reader = new BufferedReader(new InputStreamReader(fis))) {
            
            String line = reader.readLine();
            System.out.println(line);
            
            // Resources automatically closed even if exception occurs
        } catch (IOException e) {
            System.out.println("IO Error: " + e.getMessage());
        }
        // No finally block needed for resource cleanup
    }
    
    // Multiple resources
    public void multipleResources() {
        try (Scanner scanner = new Scanner(System.in);
             FileWriter writer = new FileWriter("output.txt");
             PrintWriter printWriter = new PrintWriter(writer)) {
            
            System.out.print("Enter text: ");
            String input = scanner.nextLine();
            printWriter.println(input);
            
        } catch (IOException e) {
            System.out.println("IO Error: " + e.getMessage());
        }
    }
    
    // Custom resource implementing AutoCloseable
    static class CustomResource implements AutoCloseable {
        private String name;
        
        public CustomResource(String name) {
            this.name = name;
            System.out.println("Opening resource: " + name);
        }
        
        public void doSomething() {
            System.out.println("Using resource: " + name);
        }
        
        @Override
        public void close() {
            System.out.println("Closing resource: " + name);
        }
    }
    
    public void customResourceExample() {
        try (CustomResource resource1 = new CustomResource("Resource1");
             CustomResource resource2 = new CustomResource("Resource2")) {
            
            resource1.doSomething();
            resource2.doSomething();
            
            // Simulate exception
            throw new RuntimeException("Simulated exception");
            
        } catch (Exception e) {
            System.out.println("Exception caught: " + e.getMessage());
        }
        // Resources closed in reverse order (Resource2, then Resource1)
    }
    
    public static void main(String[] args) {
        TryWithResources example = new TryWithResources();
        
        System.out.println("=== Old Way ===");
        example.oldWayResourceManagement();
        
        System.out.println("\n=== New Way ===");
        example.newWayResourceManagement();
        
        System.out.println("\n=== Custom Resource ===");
        example.customResourceExample();
    }
}
```

### Suppressed Exceptions
```java
public class SuppressedExceptions {
    
    static class ProblematicResource implements AutoCloseable {
        private String name;
        
        public ProblematicResource(String name) {
            this.name = name;
        }
        
        public void doWork() throws Exception {
            throw new Exception("Exception from doWork() in " + name);
        }
        
        @Override
        public void close() throws Exception {
            throw new Exception("Exception from close() in " + name);
        }
    }
    
    public void demonstrateSuppressedExceptions() {
        try (ProblematicResource resource = new ProblematicResource("TestResource")) {
            resource.doWork();
        } catch (Exception e) {
            System.out.println("Primary exception: " + e.getMessage());
            
            // Check for suppressed exceptions
            Throwable[] suppressed = e.getSuppressed();
            for (Throwable t : suppressed) {
                System.out.println("Suppressed exception: " + t.getMessage());
            }
        }
    }
    
    public static void main(String[] args) {
        new SuppressedExceptions().demonstrateSuppressedExceptions();
    }
}
```

## 4. Throwing and Creating Custom Exceptions

### Throwing Exceptions
```java
public class ThrowingExceptions {
    
    public void validateAge(int age) {
        if (age < 0) {
            throw new IllegalArgumentException("Age cannot be negative: " + age);
        }
        if (age > 150) {
            throw new IllegalArgumentException("Age cannot be greater than 150: " + age);
        }
        System.out.println("Valid age: " + age);
    }
    
    public void demonstrateThrows() throws IOException, InterruptedException {
        // Method declares that it might throw these exceptions
        if (Math.random() > 0.7) {
            throw new IOException("Random IO exception");
        }
        if (Math.random() > 0.8) {
            throw new InterruptedException("Random interruption");
        }
        System.out.println("Method completed successfully");
    }
    
    public void rethrowException() throws Exception {
        try {
            int result = 10 / 0;
        } catch (ArithmeticException e) {
            System.out.println("Caught arithmetic exception, rethrowing as general exception");
            throw new Exception("Rethrown exception", e);
        }
    }
    
    public static void main(String[] args) {
        ThrowingExceptions example = new ThrowingExceptions();
        
        // Test validateAge
        try {
            example.validateAge(25);
            example.validateAge(-5);
        } catch (IllegalArgumentException e) {
            System.out.println("Validation error: " + e.getMessage());
        }
        
        // Test method with throws clause
        try {
            example.demonstrateThrows();
        } catch (IOException e) {
            System.out.println("IO Exception: " + e.getMessage());
        } catch (InterruptedException e) {
            System.out.println("Interrupted: " + e.getMessage());
        }
        
        // Test rethrowing
        try {
            example.rethrowException();
        } catch (Exception e) {
            System.out.println("Caught rethrown exception: " + e.getMessage());
            System.out.println("Original cause: " + e.getCause().getMessage());
        }
    }
}
```

### Custom Exception Classes
```java
// Custom checked exception
class InvalidAccountException extends Exception {
    private String accountNumber;
    
    public InvalidAccountException(String message) {
        super(message);
    }
    
    public InvalidAccountException(String message, String accountNumber) {
        super(message);
        this.accountNumber = accountNumber;
    }
    
    public InvalidAccountException(String message, Throwable cause) {
        super(message, cause);
    }
    
    public String getAccountNumber() {
        return accountNumber;
    }
}

// Custom unchecked exception
class InsufficientFundsException extends RuntimeException {
    private double requestedAmount;
    private double availableAmount;
    
    public InsufficientFundsException(String message, double requestedAmount, double availableAmount) {
        super(message);
        this.requestedAmount = requestedAmount;
        this.availableAmount = availableAmount;
    }
    
    public double getRequestedAmount() {
        return requestedAmount;
    }
    
    public double getAvailableAmount() {
        return availableAmount;
    }
    
    @Override
    public String toString() {
        return super.toString() + 
               " [Requested: " + requestedAmount + 
               ", Available: " + availableAmount + "]";
    }
}

// Business logic class using custom exceptions
class BankAccount {
    private String accountNumber;
    private double balance;
    
    public BankAccount(String accountNumber, double initialBalance) throws InvalidAccountException {
        if (accountNumber == null || accountNumber.trim().isEmpty()) {
            throw new InvalidAccountException("Account number cannot be null or empty", accountNumber);
        }
        if (initialBalance < 0) {
            throw new InvalidAccountException("Initial balance cannot be negative");
        }
        
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }
    
    public void withdraw(double amount) throws InsufficientFundsException {
        if (amount <= 0) {
            throw new IllegalArgumentException("Withdrawal amount must be positive");
        }
        
        if (amount > balance) {
            throw new InsufficientFundsException(
                "Insufficient funds for withdrawal", 
                amount, 
                balance
            );
        }
        
        balance -= amount;
        System.out.println("Withdrawn: $" + amount + ", New balance: $" + balance);
    }
    
    public void deposit(double amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException("Deposit amount must be positive");
        }
        
        balance += amount;
        System.out.println("Deposited: $" + amount + ", New balance: $" + balance);
    }
    
    public double getBalance() {
        return balance;
    }
    
    public String getAccountNumber() {
        return accountNumber;
    }
}

public class CustomExceptionsExample {
    public static void main(String[] args) {
        try {
            // Valid account creation
            BankAccount account = new BankAccount("ACC123", 1000.0);
            
            // Valid operations
            account.deposit(500.0);
            account.withdraw(200.0);
            
            // This will throw InsufficientFundsException
            account.withdraw(2000.0);
            
        } catch (InvalidAccountException e) {
            System.out.println("Invalid account: " + e.getMessage());
            if (e.getAccountNumber() != null) {
                System.out.println("Account number: " + e.getAccountNumber());
            }
        } catch (InsufficientFundsException e) {
            System.out.println("Transaction failed: " + e.getMessage());
            System.out.println("Requested: $" + e.getRequestedAmount());
            System.out.println("Available: $" + e.getAvailableAmount());
        } catch (IllegalArgumentException e) {
            System.out.println("Invalid argument: " + e.getMessage());
        }
        
        // Invalid account creation
        try {
            BankAccount invalidAccount = new BankAccount("", -100);
        } catch (InvalidAccountException e) {
            System.out.println("Cannot create account: " + e.getMessage());
        }
    }
}
```

## 5. Exception Handling Best Practices

### Good Exception Handling Practices
```java
import java.io.*;
import java.util.logging.Logger;
import java.util.logging.Level;

public class ExceptionBestPractices {
    private static final Logger LOGGER = Logger.getLogger(ExceptionBestPractices.class.getName());
    
    // 1. Catch specific exceptions, not generic Exception
    public void goodSpecificCatch() {
        try {
            String number = "abc";
            Integer.parseInt(number);
        } catch (NumberFormatException e) {
            // Handle specific exception
            LOGGER.log(Level.WARNING, "Invalid number format: " + e.getMessage());
        }
    }
    
    public void badGenericCatch() {
        try {
            String number = "abc";
            Integer.parseInt(number);
        } catch (Exception e) {
            // Too generic - might catch unexpected exceptions
            LOGGER.log(Level.WARNING, "Something went wrong: " + e.getMessage());
        }
    }
    
    // 2. Don't ignore exceptions
    public void badIgnoreException() {
        try {
            String number = "abc";
            Integer.parseInt(number);
        } catch (NumberFormatException e) {
            // BAD: Ignoring exception
        }
    }
    
    public void goodLogException() {
        try {
            String number = "abc";
            Integer.parseInt(number);
        } catch (NumberFormatException e) {
            // GOOD: Log the exception
            LOGGER.log(Level.WARNING, "Failed to parse number", e);
        }
    }
    
    // 3. Use finally for cleanup, or better - try-with-resources
    public void goodCleanup() {
        try (BufferedReader reader = new BufferedReader(new FileReader("file.txt"))) {
            // Read file
            String line = reader.readLine();
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Error reading file", e);
        }
        // Resources automatically closed
    }
    
    // 4. Don't use exceptions for control flow
    public void badControlFlow() {
        String[] items = {"a", "b", "c"};
        try {
            for (int i = 0; ; i++) {
                System.out.println(items[i]);
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            // BAD: Using exception to break loop
        }
    }
    
    public void goodControlFlow() {
        String[] items = {"a", "b", "c"};
        for (int i = 0; i < items.length; i++) {
            System.out.println(items[i]);
        }
    }
    
    // 5. Provide meaningful error messages
    public void validateInput(String email) throws IllegalArgumentException {
        if (email == null) {
            throw new IllegalArgumentException("Email cannot be null");
        }
        if (email.trim().isEmpty()) {
            throw new IllegalArgumentException("Email cannot be empty");
        }
        if (!email.contains("@")) {
            throw new IllegalArgumentException("Email must contain @ symbol: " + email);
        }
    }
    
    // 6. Document exceptions with @throws
    /**
     * Reads content from a file.
     * 
     * @param filename the name of the file to read
     * @return the content of the file
     * @throws FileNotFoundException if the file doesn't exist
     * @throws IOException if an I/O error occurs while reading
     * @throws IllegalArgumentException if filename is null or empty
     */
    public String readFile(String filename) throws IOException {
        if (filename == null || filename.trim().isEmpty()) {
            throw new IllegalArgumentException("Filename cannot be null or empty");
        }
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            StringBuilder content = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line).append("\n");
            }
            return content.toString();
        }
    }
    
    // 7. Wrap exceptions appropriately
    public void processData(String data) throws DataProcessingException {
        try {
            // Some complex processing
            if (data == null) {
                throw new NullPointerException("Data is null");
            }
            // More processing...
            
        } catch (NullPointerException | IllegalStateException e) {
            // Wrap low-level exceptions in business exception
            throw new DataProcessingException("Failed to process data", e);
        }
    }
    
    public static void main(String[] args) {
        ExceptionBestPractices example = new ExceptionBestPractices();
        
        example.goodSpecificCatch();
        example.goodLogException();
        example.goodCleanup();
        example.goodControlFlow();
        
        try {
            example.validateInput("invalid-email");
        } catch (IllegalArgumentException e) {
            System.out.println("Validation error: " + e.getMessage());
        }
        
        try {
            example.readFile("nonexistent.txt");
        } catch (IOException e) {
            System.out.println("File error: " + e.getMessage());
        }
        
        try {
            example.processData(null);
        } catch (DataProcessingException e) {
            System.out.println("Processing error: " + e.getMessage());
            System.out.println("Caused by: " + e.getCause().getClass().getSimpleName());
        }
    }
}

// Custom business exception
class DataProcessingException extends Exception {
    public DataProcessingException(String message) {
        super(message);
    }
    
    public DataProcessingException(String message, Throwable cause) {
        super(message, cause);
    }
}
```

### Exception Handling Patterns
```java
import java.util.*;
import java.util.function.Function;

public class ExceptionHandlingPatterns {
    
    // 1. Convert checked exceptions to unchecked
    public static <T, R> Function<T, R> unchecked(CheckedFunction<T, R> function) {
        return t -> {
            try {
                return function.apply(t);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }
    
    @FunctionalInterface
    interface CheckedFunction<T, R> {
        R apply(T t) throws Exception;
    }
    
    // 2. Optional-based error handling
    public static Optional<String> safeParseInt(String str) {
        try {
            Integer.parseInt(str);
            return Optional.of(str);
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }
    
    // 3. Result wrapper pattern
    public static class Result<T> {
        private final T value;
        private final Exception error;
        
        private Result(T value, Exception error) {
            this.value = value;
            this.error = error;
        }
        
        public static <T> Result<T> success(T value) {
            return new Result<>(value, null);
        }
        
        public static <T> Result<T> failure(Exception error) {
            return new Result<>(null, error);
        }
        
        public boolean isSuccess() {
            return error == null;
        }
        
        public boolean isFailure() {
            return error != null;
        }
        
        public T getValue() {
            if (isFailure()) {
                throw new RuntimeException("Cannot get value from failed result", error);
            }
            return value;
        }
        
        public Exception getError() {
            return error;
        }
        
        public T getOrElse(T defaultValue) {
            return isSuccess() ? value : defaultValue;
        }
        
        public <U> Result<U> map(Function<T, U> mapper) {
            if (isFailure()) {
                return Result.failure(error);
            }
            try {
                return Result.success(mapper.apply(value));
            } catch (Exception e) {
                return Result.failure(e);
            }
        }
    }
    
    public static Result<Integer> safeDivide(int a, int b) {
        try {
            if (b == 0) {
                throw new ArithmeticException("Division by zero");
            }
            return Result.success(a / b);
        } catch (Exception e) {
            return Result.failure(e);
        }
    }
    
    // 4. Chain of responsibility for exception handling
    public static abstract class ExceptionHandler {
        private ExceptionHandler next;
        
        public void setNext(ExceptionHandler next) {
            this.next = next;
        }
        
        public void handle(Exception e) {
            if (canHandle(e)) {
                doHandle(e);
            } else if (next != null) {
                next.handle(e);
            } else {
                throw new RuntimeException("Unhandled exception", e);
            }
        }
        
        protected abstract boolean canHandle(Exception e);
        protected abstract void doHandle(Exception e);
    }
    
    public static class NullPointerExceptionHandler extends ExceptionHandler {
        @Override
        protected boolean canHandle(Exception e) {
            return e instanceof NullPointerException;
        }
        
        @Override
        protected void doHandle(Exception e) {
            System.out.println("Handling null pointer: " + e.getMessage());
        }
    }
    
    public static class IllegalArgumentExceptionHandler extends ExceptionHandler {
        @Override
        protected boolean canHandle(Exception e) {
            return e instanceof IllegalArgumentException;
        }
        
        @Override
        protected void doHandle(Exception e) {
            System.out.println("Handling illegal argument: " + e.getMessage());
        }
    }
    
    public static class GenericExceptionHandler extends ExceptionHandler {
        @Override
        protected boolean canHandle(Exception e) {
            return true; // Handle any exception
        }
        
        @Override
        protected void doHandle(Exception e) {
            System.out.println("Generic handler: " + e.getClass().getSimpleName() + " - " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        // 1. Unchecked function example
        List<String> numbers = Arrays.asList("1", "2", "abc", "4");
        
        numbers.stream()
               .map(unchecked(s -> Integer.parseInt(s)))
               .forEach(System.out::println);
        
        // 2. Optional-based error handling
        Optional<String> result = safeParseInt("123");
        result.ifPresentOrElse(
            s -> System.out.println("Parsed successfully: " + s),
            () -> System.out.println("Failed to parse")
        );
        
        // 3. Result wrapper pattern
        Result<Integer> divisionResult = safeDivide(10, 2);
        if (divisionResult.isSuccess()) {
            System.out.println("Division result: " + divisionResult.getValue());
        } else {
            System.out.println("Division failed: " + divisionResult.getError().getMessage());
        }
        
        Result<Integer> chainedResult = safeDivide(20, 4)
            .map(x -> x * 2)
            .map(x -> x + 10);
        
        System.out.println("Chained result: " + chainedResult.getOrElse(-1));
        
        // 4. Chain of responsibility
        ExceptionHandler handler = new NullPointerExceptionHandler();
        handler.setNext(new IllegalArgumentExceptionHandler());
        
        ExceptionHandler genericHandler = new GenericExceptionHandler();
        handler.setNext(genericHandler);
        
        // Test different exceptions
        handler.handle(new NullPointerException("Test null pointer"));
        handler.handle(new IllegalArgumentException("Test illegal argument"));
        handler.handle(new RuntimeException("Test runtime exception"));
    }
}
```

## 6. Exception Handling in Modern Java

### Exception Handling with Streams
```java
import java.util.*;
import java.util.stream.*;

public class ExceptionsInStreams {
    
    // Utility method to handle checked exceptions in streams
    public static <T, R> Function<T, R> wrapException(CheckedFunction<T, R> function) {
        return t -> {
            try {
                return function.apply(t);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }
    
    @FunctionalInterface
    interface CheckedFunction<T, R> {
        R apply(T t) throws Exception;
    }
    
    // Safe mapping that filters out failures
    public static <T, R> Function<T, Optional<R>> safe(Function<T, R> function) {
        return t -> {
            try {
                return Optional.of(function.apply(t));
            } catch (Exception e) {
                return Optional.empty();
            }
        };
    }
    
    public static void main(String[] args) {
        List<String> numbers = Arrays.asList("1", "2", "abc", "4", "xyz", "5");
        
        // 1. Handle exceptions with safe mapping
        List<Integer> validNumbers = numbers.stream()
            .map(safe(Integer::parseInt))
            .filter(Optional::isPresent)
            .map(Optional::get)
            .collect(Collectors.toList());
        
        System.out.println("Valid numbers: " + validNumbers);
        
        // 2. Collect successes and failures separately
        Map<Boolean, List<String>> partitioned = numbers.stream()
            .collect(Collectors.partitioningBy(s -> {
                try {
                    Integer.parseInt(s);
                    return true;
                } catch (NumberFormatException e) {
                    return false;
                }
            }));
        
        System.out.println("Valid strings: " + partitioned.get(true));
        System.out.println("Invalid strings: " + partitioned.get(false));
        
        // 3. Use flatMap with Optional to handle exceptions
        List<Integer> parsedNumbers = numbers.stream()
            .flatMap(s -> {
                try {
                    return Stream.of(Integer.parseInt(s));
                } catch (NumberFormatException e) {
                    return Stream.empty();
                }
            })
            .collect(Collectors.toList());
        
        System.out.println("Parsed numbers: " + parsedNumbers);
    }
}
```

### Exception Handling with CompletableFuture
```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class ExceptionsInCompletableFuture {
    
    public static void main(String[] args) {
        // 1. Handle exceptions with handle()
        CompletableFuture<String> future1 = CompletableFuture
            .supplyAsync(() -> {
                if (Math.random() > 0.5) {
                    throw new RuntimeException("Random error");
                }
                return "Success";
            })
            .handle((result, throwable) -> {
                if (throwable != null) {
                    return "Error: " + throwable.getMessage();
                }
                return result;
            });
        
        future1.thenAccept(System.out::println);
        
        // 2. Handle exceptions with exceptionally()
        CompletableFuture<String> future2 = CompletableFuture
            .supplyAsync(() -> {
                throw new RuntimeException("Simulated error");
            })
            .exceptionally(throwable -> {
                System.out.println("Exception caught: " + throwable.getMessage());
                return "Default value";
            });
        
        future2.thenAccept(result -> System.out.println("Result: " + result));
        
        // 3. Handle exceptions with whenComplete()
        CompletableFuture<String> future3 = CompletableFuture
            .supplyAsync(() -> {
                if (Math.random() > 0.5) {
                    throw new RuntimeException("Another error");
                }
                return "Another success";
            })
            .whenComplete((result, throwable) -> {
                if (throwable != null) {
                    System.out.println("Task failed: " + throwable.getMessage());
                } else {
                    System.out.println("Task succeeded: " + result);
                }
            });
        
        // 4. Chain operations with exception handling
        CompletableFuture<Integer> future4 = CompletableFuture
            .supplyAsync(() -> "42")
            .thenApply(s -> {
                if (s.equals("42")) {
                    throw new RuntimeException("Don't like 42");
                }
                return Integer.parseInt(s);
            })
            .handle((result, throwable) -> {
                if (throwable != null) {
                    return 0; // Default value
                }
                return result;
            });
        
        try {
            System.out.println("Final result: " + future4.get());
        } catch (InterruptedException | ExecutionException e) {
            System.out.println("Error getting result: " + e.getMessage());
        }
        
        // Wait for async operations to complete
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

## Summary

Exception handling in Java provides:

- **Structured Error Handling**: try-catch-finally blocks for organized error management
- **Exception Hierarchy**: Checked vs unchecked exceptions with clear inheritance
- **Resource Management**: try-with-resources for automatic cleanup
- **Custom Exceptions**: Creating domain-specific exception types
- **Best Practices**: Guidelines for effective exception handling

Key principles:
- Handle specific exceptions, not generic ones
- Don't ignore exceptions - log or handle appropriately
- Use try-with-resources for automatic resource management
- Don't use exceptions for control flow
- Provide meaningful error messages
- Document exceptions in method signatures
- Wrap exceptions appropriately for abstraction layers
- Consider modern patterns like Optional and Result types for error handling

Exception handling is crucial for:
- Robust application behavior
- Proper resource cleanup
- Clear error communication
- Maintainable code
- User experience
