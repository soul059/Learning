# I/O and NIO in Java

## 1. Traditional I/O (java.io)

### File Operations
```java
import java.io.*;
import java.util.Scanner;

public class FileOperations {
    
    public void createAndWriteFile() {
        // Create file and write content
        try {
            File file = new File("example.txt");
            
            // Create file if it doesn't exist
            if (file.createNewFile()) {
                System.out.println("File created: " + file.getName());
            } else {
                System.out.println("File already exists");
            }
            
            // Write to file using FileWriter
            try (FileWriter writer = new FileWriter(file)) {
                writer.write("Hello, World!\n");
                writer.write("This is a test file.\n");
                writer.write("Java I/O operations.");
            }
            
            System.out.println("Content written to file");
            
        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
    
    public void readFile() {
        // Different ways to read a file
        
        // 1. Using FileReader
        try (FileReader reader = new FileReader("example.txt")) {
            int character;
            System.out.println("Reading with FileReader:");
            while ((character = reader.read()) != -1) {
                System.out.print((char) character);
            }
            System.out.println();
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        }
        
        // 2. Using BufferedReader (more efficient)
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader("example.txt"))) {
            String line;
            System.out.println("\nReading with BufferedReader:");
            while ((line = bufferedReader.readLine()) != null) {
                System.out.println(line);
            }
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        }
        
        // 3. Using Scanner
        try (Scanner scanner = new Scanner(new File("example.txt"))) {
            System.out.println("\nReading with Scanner:");
            while (scanner.hasNextLine()) {
                System.out.println(scanner.nextLine());
            }
        } catch (FileNotFoundException e) {
            System.out.println("File not found: " + e.getMessage());
        }
    }
    
    public void fileInformation() {
        File file = new File("example.txt");
        
        if (file.exists()) {
            System.out.println("File Information:");
            System.out.println("Name: " + file.getName());
            System.out.println("Absolute Path: " + file.getAbsolutePath());
            System.out.println("Parent: " + file.getParent());
            System.out.println("Size: " + file.length() + " bytes");
            System.out.println("Last Modified: " + new java.util.Date(file.lastModified()));
            System.out.println("Is Directory: " + file.isDirectory());
            System.out.println("Is File: " + file.isFile());
            System.out.println("Can Read: " + file.canRead());
            System.out.println("Can Write: " + file.canWrite());
            System.out.println("Can Execute: " + file.canExecute());
        } else {
            System.out.println("File does not exist");
        }
    }
    
    public void directoryOperations() {
        // Create directory
        File directory = new File("testDirectory");
        if (directory.mkdir()) {
            System.out.println("Directory created: " + directory.getName());
        }
        
        // Create nested directories
        File nestedDir = new File("parent/child/grandchild");
        if (nestedDir.mkdirs()) {
            System.out.println("Nested directories created");
        }
        
        // List files in current directory
        File currentDir = new File(".");
        File[] files = currentDir.listFiles();
        
        if (files != null) {
            System.out.println("\nFiles in current directory:");
            for (File file : files) {
                if (file.isDirectory()) {
                    System.out.println("[DIR] " + file.getName());
                } else {
                    System.out.println("[FILE] " + file.getName() + " (" + file.length() + " bytes)");
                }
            }
        }
        
        // Filter files
        File[] txtFiles = currentDir.listFiles((dir, name) -> name.endsWith(".txt"));
        if (txtFiles != null) {
            System.out.println("\nText files:");
            for (File file : txtFiles) {
                System.out.println(file.getName());
            }
        }
    }
    
    public static void main(String[] args) {
        FileOperations fileOps = new FileOperations();
        
        fileOps.createAndWriteFile();
        fileOps.readFile();
        fileOps.fileInformation();
        fileOps.directoryOperations();
    }
}
```

### Byte Streams
```java
import java.io.*;

public class ByteStreams {
    
    public void fileInputOutputStream() {
        // Writing bytes to file
        try (FileOutputStream fos = new FileOutputStream("bytes.dat")) {
            byte[] data = "Hello World in bytes!".getBytes();
            fos.write(data);
            
            // Write individual bytes
            fos.write(65); // 'A'
            fos.write(66); // 'B'
            fos.write(67); // 'C'
            
            System.out.println("Data written to bytes.dat");
            
        } catch (IOException e) {
            System.out.println("Error writing file: " + e.getMessage());
        }
        
        // Reading bytes from file
        try (FileInputStream fis = new FileInputStream("bytes.dat")) {
            int bytesRead;
            System.out.println("Reading bytes:");
            
            while ((bytesRead = fis.read()) != -1) {
                System.out.print((char) bytesRead);
            }
            System.out.println();
            
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        }
    }
    
    public void bufferedByteStreams() {
        // Buffered streams for better performance
        try (BufferedOutputStream bos = new BufferedOutputStream(
                new FileOutputStream("buffered.dat"))) {
            
            String data = "This is buffered output stream example.";
            bos.write(data.getBytes());
            bos.flush(); // Ensure data is written
            
        } catch (IOException e) {
            System.out.println("Error with buffered output: " + e.getMessage());
        }
        
        try (BufferedInputStream bis = new BufferedInputStream(
                new FileInputStream("buffered.dat"))) {
            
            byte[] buffer = new byte[1024];
            int bytesRead = bis.read(buffer);
            
            if (bytesRead > 0) {
                String content = new String(buffer, 0, bytesRead);
                System.out.println("Buffered read: " + content);
            }
            
        } catch (IOException e) {
            System.out.println("Error with buffered input: " + e.getMessage());
        }
    }
    
    public void copyFile(String source, String destination) {
        try (FileInputStream fis = new FileInputStream(source);
             FileOutputStream fos = new FileOutputStream(destination)) {
            
            byte[] buffer = new byte[4096]; // 4KB buffer
            int bytesRead;
            
            while ((bytesRead = fis.read(buffer)) != -1) {
                fos.write(buffer, 0, bytesRead);
            }
            
            System.out.println("File copied from " + source + " to " + destination);
            
        } catch (IOException e) {
            System.out.println("Error copying file: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        ByteStreams byteStreams = new ByteStreams();
        
        byteStreams.fileInputOutputStream();
        byteStreams.bufferedByteStreams();
        
        // Copy a file (make sure source exists)
        // byteStreams.copyFile("source.txt", "destination.txt");
    }
}
```

### Character Streams
```java
import java.io.*;
import java.nio.charset.StandardCharsets;

public class CharacterStreams {
    
    public void fileReaderWriter() {
        // Writing characters to file
        try (FileWriter writer = new FileWriter("characters.txt", StandardCharsets.UTF_8)) {
            writer.write("Hello World with characters!\n");
            writer.write("支持Unicode字符\n");
            writer.write("Emoji: 😀🎉🚀\n");
            
            // Write character array
            char[] chars = {'J', 'a', 'v', 'a', '\n'};
            writer.write(chars);
            
        } catch (IOException e) {
            System.out.println("Error writing characters: " + e.getMessage());
        }
        
        // Reading characters from file
        try (FileReader reader = new FileReader("characters.txt", StandardCharsets.UTF_8)) {
            int character;
            System.out.println("Reading characters:");
            
            while ((character = reader.read()) != -1) {
                System.out.print((char) character);
            }
            
        } catch (IOException e) {
            System.out.println("Error reading characters: " + e.getMessage());
        }
    }
    
    public void bufferedCharacterStreams() {
        // Writing with BufferedWriter
        try (BufferedWriter writer = new BufferedWriter(
                new FileWriter("buffered_chars.txt"))) {
            
            writer.write("Line 1");
            writer.newLine();
            writer.write("Line 2");
            writer.newLine();
            writer.write("Line 3");
            
        } catch (IOException e) {
            System.out.println("Error with buffered writer: " + e.getMessage());
        }
        
        // Reading with BufferedReader
        try (BufferedReader reader = new BufferedReader(
                new FileReader("buffered_chars.txt"))) {
            
            String line;
            System.out.println("\nBuffered character reading:");
            int lineNumber = 1;
            
            while ((line = reader.readLine()) != null) {
                System.out.println(lineNumber + ": " + line);
                lineNumber++;
            }
            
        } catch (IOException e) {
            System.out.println("Error with buffered reader: " + e.getMessage());
        }
    }
    
    public void printWriterExample() {
        // PrintWriter provides convenient methods
        try (PrintWriter writer = new PrintWriter(
                new FileWriter("print_output.txt"))) {
            
            writer.println("Hello, PrintWriter!");
            writer.printf("Formatted number: %.2f%n", 123.456);
            writer.printf("Name: %s, Age: %d%n", "John", 25);
            
            // Print different data types
            writer.println(42);
            writer.println(true);
            writer.println(3.14159);
            
        } catch (IOException e) {
            System.out.println("Error with PrintWriter: " + e.getMessage());
        }
        
        // Reading the output
        try (BufferedReader reader = new BufferedReader(
                new FileReader("print_output.txt"))) {
            
            String line;
            System.out.println("\nPrintWriter output:");
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
            
        } catch (IOException e) {
            System.out.println("Error reading PrintWriter output: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        CharacterStreams charStreams = new CharacterStreams();
        
        charStreams.fileReaderWriter();
        charStreams.bufferedCharacterStreams();
        charStreams.printWriterExample();
    }
}
```

### Object Serialization
```java
import java.io.*;
import java.util.ArrayList;
import java.util.List;

// Serializable class
class Person implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private String name;
    private int age;
    private transient String password; // transient field won't be serialized
    private static String company = "TechCorp"; // static fields aren't serialized
    
    public Person(String name, int age, String password) {
        this.name = name;
        this.age = age;
        this.password = password;
    }
    
    // Getters and setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public int getAge() { return age; }
    public void setAge(int age) { this.age = age; }
    
    public String getPassword() { return password; }
    public void setPassword(String password) { this.password = password; }
    
    public static String getCompany() { return company; }
    public static void setCompany(String company) { Person.company = company; }
    
    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + 
               ", password='" + password + "', company='" + company + "'}";
    }
    
    // Custom serialization (optional)
    private void writeObject(ObjectOutputStream oos) throws IOException {
        oos.defaultWriteObject();
        // Custom serialization logic
        oos.writeObject("Custom data");
    }
    
    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        ois.defaultReadObject();
        // Custom deserialization logic
        String customData = (String) ois.readObject();
        System.out.println("Custom data read: " + customData);
    }
}

public class ObjectSerialization {
    
    public void serializeObjects() {
        List<Person> people = new ArrayList<>();
        people.add(new Person("Alice", 30, "secret123"));
        people.add(new Person("Bob", 25, "password456"));
        people.add(new Person("Charlie", 35, "admin789"));
        
        // Serialize objects to file
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream("people.ser"))) {
            
            oos.writeObject(people);
            System.out.println("Objects serialized successfully");
            
        } catch (IOException e) {
            System.out.println("Error serializing objects: " + e.getMessage());
        }
    }
    
    @SuppressWarnings("unchecked")
    public void deserializeObjects() {
        // Deserialize objects from file
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream("people.ser"))) {
            
            List<Person> people = (List<Person>) ois.readObject();
            
            System.out.println("Objects deserialized successfully:");
            for (Person person : people) {
                System.out.println(person);
            }
            
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("Error deserializing objects: " + e.getMessage());
        }
    }
    
    public void serializeSingleObject() {
        Person person = new Person("John", 28, "mypassword");
        
        // Serialize single object
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream("person.ser"))) {
            
            oos.writeObject(person);
            System.out.println("Single object serialized");
            
        } catch (IOException e) {
            System.out.println("Error serializing object: " + e.getMessage());
        }
        
        // Deserialize single object
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream("person.ser"))) {
            
            Person deserializedPerson = (Person) ois.readObject();
            System.out.println("Deserialized person: " + deserializedPerson);
            
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("Error deserializing object: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        ObjectSerialization serialization = new ObjectSerialization();
        
        serialization.serializeObjects();
        serialization.deserializeObjects();
        
        System.out.println();
        serialization.serializeSingleObject();
    }
}
```

## 2. New I/O (NIO) - java.nio

### Path and Files
```java
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;

public class PathAndFiles {
    
    public void pathOperations() {
        // Creating paths
        Path path1 = Paths.get("example.txt");
        Path path2 = Paths.get("folder", "subfolder", "file.txt");
        Path path3 = Path.of("modern", "way", "to", "create", "path.txt"); // Java 11+
        
        System.out.println("Path 1: " + path1);
        System.out.println("Path 2: " + path2);
        System.out.println("Path 3: " + path3);
        
        // Path information
        Path currentDir = Paths.get(".");
        System.out.println("\nPath Information:");
        System.out.println("Absolute path: " + currentDir.toAbsolutePath());
        System.out.println("Normalized path: " + currentDir.toAbsolutePath().normalize());
        System.out.println("Parent: " + currentDir.getParent());
        System.out.println("File name: " + path1.getFileName());
        System.out.println("Root: " + currentDir.toAbsolutePath().getRoot());
        
        // Path operations
        Path basePath = Paths.get("/home/user");
        Path fullPath = basePath.resolve("documents/file.txt");
        System.out.println("Resolved path: " + fullPath);
        
        Path relativePath = basePath.relativize(fullPath);
        System.out.println("Relative path: " + relativePath);
    }
    
    public void fileOperations() {
        Path testFile = Paths.get("nio_test.txt");
        
        try {
            // Create file
            if (!Files.exists(testFile)) {
                Files.createFile(testFile);
                System.out.println("File created: " + testFile);
            }
            
            // Write to file
            String content = "Hello NIO!\nThis is a test file.\nNIO is powerful!";
            Files.write(testFile, content.getBytes());
            System.out.println("Content written to file");
            
            // Read from file
            List<String> lines = Files.readAllLines(testFile);
            System.out.println("\nFile content:");
            lines.forEach(System.out::println);
            
            // Read as string (Java 11+)
            String fileContent = Files.readString(testFile);
            System.out.println("\nFile as string:\n" + fileContent);
            
            // File attributes
            BasicFileAttributes attrs = Files.readAttributes(testFile, BasicFileAttributes.class);
            System.out.println("\nFile attributes:");
            System.out.println("Size: " + attrs.size() + " bytes");
            System.out.println("Creation time: " + attrs.creationTime());
            System.out.println("Last modified: " + attrs.lastModifiedTime());
            System.out.println("Is directory: " + attrs.isDirectory());
            System.out.println("Is regular file: " + attrs.isRegularFile());
            
        } catch (IOException e) {
            System.out.println("Error with file operations: " + e.getMessage());
        }
    }
    
    public void directoryOperations() {
        Path testDir = Paths.get("test_directory");
        
        try {
            // Create directory
            if (!Files.exists(testDir)) {
                Files.createDirectory(testDir);
                System.out.println("Directory created: " + testDir);
            }
            
            // Create directories with parents
            Path nestedDir = Paths.get("parent/child/grandchild");
            Files.createDirectories(nestedDir);
            System.out.println("Nested directories created");
            
            // List directory contents
            System.out.println("\nCurrent directory contents:");
            try (Stream<Path> paths = Files.list(Paths.get("."))) {
                paths.filter(Files::isRegularFile)
                     .forEach(System.out::println);
            }
            
            // Walk directory tree
            System.out.println("\nWalking directory tree:");
            try (Stream<Path> paths = Files.walk(Paths.get("."))) {
                paths.filter(Files::isRegularFile)
                     .filter(path -> path.toString().endsWith(".txt"))
                     .forEach(System.out::println);
            }
            
        } catch (IOException e) {
            System.out.println("Error with directory operations: " + e.getMessage());
        }
    }
    
    public void copyAndMoveOperations() {
        Path source = Paths.get("nio_test.txt");
        Path copyDestination = Paths.get("nio_copy.txt");
        Path moveDestination = Paths.get("nio_moved.txt");
        
        try {
            // Copy file
            Files.copy(source, copyDestination, StandardCopyOption.REPLACE_EXISTING);
            System.out.println("File copied to " + copyDestination);
            
            // Move file
            Files.move(copyDestination, moveDestination, StandardCopyOption.REPLACE_EXISTING);
            System.out.println("File moved to " + moveDestination);
            
            // Delete file
            Files.delete(moveDestination);
            System.out.println("File deleted: " + moveDestination);
            
        } catch (IOException e) {
            System.out.println("Error with copy/move operations: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        PathAndFiles pathFiles = new PathAndFiles();
        
        pathFiles.pathOperations();
        pathFiles.fileOperations();
        pathFiles.directoryOperations();
        pathFiles.copyAndMoveOperations();
    }
}
```

### Channels and Buffers
```java
import java.nio.*;
import java.nio.channels.*;
import java.io.*;

public class ChannelsAndBuffers {
    
    public void bufferOperations() {
        // Create different types of buffers
        ByteBuffer byteBuffer = ByteBuffer.allocate(1024);
        CharBuffer charBuffer = CharBuffer.allocate(1024);
        IntBuffer intBuffer = IntBuffer.allocate(100);
        
        System.out.println("Buffer Operations:");
        
        // Write data to buffer
        String message = "Hello NIO Buffers!";
        byteBuffer.put(message.getBytes());
        
        System.out.println("After writing - Position: " + byteBuffer.position() + 
                          ", Limit: " + byteBuffer.limit() + 
                          ", Capacity: " + byteBuffer.capacity());
        
        // Flip buffer for reading
        byteBuffer.flip();
        System.out.println("After flip - Position: " + byteBuffer.position() + 
                          ", Limit: " + byteBuffer.limit());
        
        // Read data from buffer
        byte[] data = new byte[byteBuffer.remaining()];
        byteBuffer.get(data);
        System.out.println("Read data: " + new String(data));
        
        // Clear buffer for reuse
        byteBuffer.clear();
        System.out.println("After clear - Position: " + byteBuffer.position() + 
                          ", Limit: " + byteBuffer.limit());
        
        // Demonstrate mark and reset
        byteBuffer.put("ABCDEFGH".getBytes());
        byteBuffer.flip();
        
        System.out.println("\nMark and Reset:");
        System.out.println("First char: " + (char) byteBuffer.get());
        System.out.println("Second char: " + (char) byteBuffer.get());
        
        byteBuffer.mark(); // Mark current position
        System.out.println("Third char: " + (char) byteBuffer.get());
        System.out.println("Fourth char: " + (char) byteBuffer.get());
        
        byteBuffer.reset(); // Reset to marked position
        System.out.println("After reset - Third char again: " + (char) byteBuffer.get());
    }
    
    public void fileChannelOperations() {
        String fileName = "channel_test.txt";
        String content = "This is a test file for NIO channels!\nLine 2\nLine 3";
        
        // Writing with FileChannel
        try (RandomAccessFile file = new RandomAccessFile(fileName, "rw");
             FileChannel channel = file.getChannel()) {
            
            ByteBuffer buffer = ByteBuffer.allocate(1024);
            buffer.put(content.getBytes());
            buffer.flip();
            
            int bytesWritten = channel.write(buffer);
            System.out.println("Bytes written: " + bytesWritten);
            
        } catch (IOException e) {
            System.out.println("Error writing with channel: " + e.getMessage());
        }
        
        // Reading with FileChannel
        try (RandomAccessFile file = new RandomAccessFile(fileName, "r");
             FileChannel channel = file.getChannel()) {
            
            ByteBuffer buffer = ByteBuffer.allocate(1024);
            int bytesRead = channel.read(buffer);
            
            if (bytesRead > 0) {
                buffer.flip();
                byte[] data = new byte[buffer.remaining()];
                buffer.get(data);
                System.out.println("\nRead from channel:");
                System.out.println(new String(data));
            }
            
        } catch (IOException e) {
            System.out.println("Error reading with channel: " + e.getMessage());
        }
    }
    
    public void transferBetweenChannels() {
        String sourceFile = "channel_test.txt";
        String destinationFile = "channel_copy.txt";
        
        try (RandomAccessFile source = new RandomAccessFile(sourceFile, "r");
             RandomAccessFile destination = new RandomAccessFile(destinationFile, "rw");
             FileChannel sourceChannel = source.getChannel();
             FileChannel destinationChannel = destination.getChannel()) {
            
            // Transfer from source to destination
            long bytesTransferred = sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
            System.out.println("Bytes transferred: " + bytesTransferred);
            
            // Alternative: transferFrom
            // destinationChannel.transferFrom(sourceChannel, 0, sourceChannel.size());
            
        } catch (IOException e) {
            System.out.println("Error transferring between channels: " + e.getMessage());
        }
    }
    
    public void memoryMappedFiles() {
        String fileName = "memory_mapped.txt";
        String content = "This file will be memory mapped for efficient access!";
        
        try (RandomAccessFile file = new RandomAccessFile(fileName, "rw");
             FileChannel channel = file.getChannel()) {
            
            // Write initial content
            channel.write(ByteBuffer.wrap(content.getBytes()));
            
            // Create memory-mapped buffer
            MappedByteBuffer mappedBuffer = channel.map(
                FileChannel.MapMode.READ_WRITE, 0, file.length()
            );
            
            // Read from mapped buffer
            byte[] data = new byte[(int) file.length()];
            mappedBuffer.get(data);
            System.out.println("Memory mapped read: " + new String(data));
            
            // Modify through mapped buffer
            mappedBuffer.position(0);
            mappedBuffer.put("MODIFIED".getBytes());
            mappedBuffer.force(); // Force changes to be written to disk
            
            System.out.println("File modified through memory mapping");
            
        } catch (IOException e) {
            System.out.println("Error with memory mapped file: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        ChannelsAndBuffers channelsBuffers = new ChannelsAndBuffers();
        
        channelsBuffers.bufferOperations();
        channelsBuffers.fileChannelOperations();
        channelsBuffers.transferBetweenChannels();
        channelsBuffers.memoryMappedFiles();
    }
}
```

### Non-blocking I/O (NIO.2)
```java
import java.nio.*;
import java.nio.channels.*;
import java.net.*;
import java.io.IOException;
import java.util.Iterator;
import java.util.Set;

public class NonBlockingIO {
    
    // Simple echo server using non-blocking I/O
    public void echoServer() {
        try {
            Selector selector = Selector.open();
            ServerSocketChannel serverChannel = ServerSocketChannel.open();
            
            // Configure server channel
            serverChannel.configureBlocking(false);
            serverChannel.bind(new InetSocketAddress(8080));
            serverChannel.register(selector, SelectionKey.OP_ACCEPT);
            
            System.out.println("Echo server started on port 8080");
            
            while (true) {
                // Wait for events
                int readyChannels = selector.select();
                
                if (readyChannels == 0) {
                    continue;
                }
                
                Set<SelectionKey> selectedKeys = selector.selectedKeys();
                Iterator<SelectionKey> keyIterator = selectedKeys.iterator();
                
                while (keyIterator.hasNext()) {
                    SelectionKey key = keyIterator.next();
                    
                    if (key.isAcceptable()) {
                        handleAccept(key, selector);
                    } else if (key.isReadable()) {
                        handleRead(key);
                    }
                    
                    keyIterator.remove();
                }
            }
            
        } catch (IOException e) {
            System.out.println("Error in echo server: " + e.getMessage());
        }
    }
    
    private void handleAccept(SelectionKey key, Selector selector) throws IOException {
        ServerSocketChannel serverChannel = (ServerSocketChannel) key.channel();
        SocketChannel clientChannel = serverChannel.accept();
        
        if (clientChannel != null) {
            clientChannel.configureBlocking(false);
            clientChannel.register(selector, SelectionKey.OP_READ);
            System.out.println("Client connected: " + clientChannel.getRemoteAddress());
        }
    }
    
    private void handleRead(SelectionKey key) throws IOException {
        SocketChannel clientChannel = (SocketChannel) key.channel();
        ByteBuffer buffer = ByteBuffer.allocate(1024);
        
        try {
            int bytesRead = clientChannel.read(buffer);
            
            if (bytesRead > 0) {
                buffer.flip();
                
                // Echo the data back
                while (buffer.hasRemaining()) {
                    clientChannel.write(buffer);
                }
                
                buffer.clear();
            } else if (bytesRead == -1) {
                // Client disconnected
                System.out.println("Client disconnected: " + clientChannel.getRemoteAddress());
                clientChannel.close();
                key.cancel();
            }
            
        } catch (IOException e) {
            System.out.println("Error reading from client: " + e.getMessage());
            clientChannel.close();
            key.cancel();
        }
    }
    
    // Simple client to test the echo server
    public void echoClient() {
        try (SocketChannel channel = SocketChannel.open()) {
            channel.connect(new InetSocketAddress("localhost", 8080));
            
            String message = "Hello from NIO client!";
            ByteBuffer buffer = ByteBuffer.wrap(message.getBytes());
            
            // Send message
            channel.write(buffer);
            
            // Read echo response
            buffer.clear();
            int bytesRead = channel.read(buffer);
            
            if (bytesRead > 0) {
                buffer.flip();
                byte[] data = new byte[buffer.remaining()];
                buffer.get(data);
                System.out.println("Echo response: " + new String(data));
            }
            
        } catch (IOException e) {
            System.out.println("Error in echo client: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        NonBlockingIO nio = new NonBlockingIO();
        
        // Start server in a separate thread
        Thread serverThread = new Thread(() -> nio.echoServer());
        serverThread.setDaemon(true);
        serverThread.start();
        
        // Give server time to start
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Test with client
        nio.echoClient();
    }
}
```

## 3. Advanced I/O Operations

### File Watching Service
```java
import java.nio.file.*;
import java.io.IOException;

public class FileWatchingService {
    
    public void watchDirectory(String directoryPath) {
        try {
            Path path = Paths.get(directoryPath);
            WatchService watchService = FileSystems.getDefault().newWatchService();
            
            // Register directory for different types of events
            path.register(watchService,
                StandardWatchEventKinds.ENTRY_CREATE,
                StandardWatchEventKinds.ENTRY_DELETE,
                StandardWatchEventKinds.ENTRY_MODIFY);
            
            System.out.println("Watching directory: " + path.toAbsolutePath());
            System.out.println("Make some changes to files in this directory...");
            
            boolean valid = true;
            while (valid) {
                WatchKey watchKey;
                
                try {
                    // Wait for events
                    watchKey = watchService.take();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
                
                // Process events
                for (WatchEvent<?> event : watchKey.pollEvents()) {
                    WatchEvent.Kind<?> kind = event.kind();
                    
                    // Skip overflow events
                    if (kind == StandardWatchEventKinds.OVERFLOW) {
                        continue;
                    }
                    
                    // Get the filename for the event
                    @SuppressWarnings("unchecked")
                    WatchEvent<Path> pathEvent = (WatchEvent<Path>) event;
                    Path filename = pathEvent.context();
                    
                    // Print event details
                    System.out.printf("Event: %s, File: %s%n", kind.name(), filename);
                    
                    // Handle specific events
                    if (kind == StandardWatchEventKinds.ENTRY_CREATE) {
                        System.out.println("  -> New file created: " + filename);
                    } else if (kind == StandardWatchEventKinds.ENTRY_DELETE) {
                        System.out.println("  -> File deleted: " + filename);
                    } else if (kind == StandardWatchEventKinds.ENTRY_MODIFY) {
                        System.out.println("  -> File modified: " + filename);
                    }
                }
                
                // Reset the watch key
                valid = watchKey.reset();
            }
            
        } catch (IOException e) {
            System.out.println("Error setting up file watcher: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        FileWatchingService watcher = new FileWatchingService();
        
        // Watch current directory
        watcher.watchDirectory(".");
    }
}
```

### Asynchronous File Operations
```java
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.CompletionHandler;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Future;

public class AsynchronousFileOperations {
    
    public void futureBasedAsyncRead() {
        try {
            AsynchronousFileChannel fileChannel = AsynchronousFileChannel.open(
                Paths.get("async_test.txt"), StandardOpenOption.READ);
            
            ByteBuffer buffer = ByteBuffer.allocate(1024);
            
            // Asynchronous read using Future
            Future<Integer> operation = fileChannel.read(buffer, 0);
            
            // Do other work while reading
            System.out.println("Reading file asynchronously...");
            
            // Get result when ready
            Integer bytesRead = operation.get();
            
            if (bytesRead > 0) {
                buffer.flip();
                byte[] data = new byte[buffer.remaining()];
                buffer.get(data);
                System.out.println("Async read result: " + new String(data));
            }
            
            fileChannel.close();
            
        } catch (Exception e) {
            System.out.println("Error in async read: " + e.getMessage());
        }
    }
    
    public void callbackBasedAsyncWrite() {
        try {
            AsynchronousFileChannel fileChannel = AsynchronousFileChannel.open(
                Paths.get("async_write_test.txt"), 
                StandardOpenOption.WRITE, 
                StandardOpenOption.CREATE, 
                StandardOpenOption.TRUNCATE_EXISTING);
            
            String content = "This is asynchronous write content!";
            ByteBuffer buffer = ByteBuffer.wrap(content.getBytes());
            
            CountDownLatch latch = new CountDownLatch(1);
            
            // Asynchronous write using CompletionHandler
            fileChannel.write(buffer, 0, buffer, new CompletionHandler<Integer, ByteBuffer>() {
                @Override
                public void completed(Integer result, ByteBuffer attachment) {
                    System.out.println("Async write completed. Bytes written: " + result);
                    latch.countDown();
                    
                    try {
                        fileChannel.close();
                    } catch (Exception e) {
                        System.out.println("Error closing file: " + e.getMessage());
                    }
                }
                
                @Override
                public void failed(Throwable exc, ByteBuffer attachment) {
                    System.out.println("Async write failed: " + exc.getMessage());
                    latch.countDown();
                    
                    try {
                        fileChannel.close();
                    } catch (Exception e) {
                        System.out.println("Error closing file: " + e.getMessage());
                    }
                }
            });
            
            // Do other work while writing
            System.out.println("Writing file asynchronously...");
            
            // Wait for completion
            latch.await();
            
        } catch (Exception e) {
            System.out.println("Error in async write: " + e.getMessage());
        }
    }
    
    public void createTestFile() {
        try {
            String content = "This is a test file for asynchronous operations.\nLine 2\nLine 3";
            Files.write(Paths.get("async_test.txt"), content.getBytes());
            System.out.println("Test file created for async operations");
        } catch (Exception e) {
            System.out.println("Error creating test file: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        AsynchronousFileOperations asyncOps = new AsynchronousFileOperations();
        
        asyncOps.createTestFile();
        asyncOps.futureBasedAsyncRead();
        asyncOps.callbackBasedAsyncWrite();
    }
}
```

## 4. Performance Comparison and Best Practices

### Performance Comparison
```java
import java.io.*;
import java.nio.file.*;
import java.time.Instant;
import java.time.Duration;

public class IOPerformanceComparison {
    
    private static final String TEST_FILE = "performance_test.txt";
    private static final int FILE_SIZE = 1024 * 1024; // 1MB
    
    public void createTestFile() {
        byte[] data = new byte[FILE_SIZE];
        // Fill with some data
        for (int i = 0; i < FILE_SIZE; i++) {
            data[i] = (byte) (i % 256);
        }
        
        try {
            Files.write(Paths.get(TEST_FILE), data);
            System.out.println("Test file created: " + FILE_SIZE + " bytes");
        } catch (IOException e) {
            System.out.println("Error creating test file: " + e.getMessage());
        }
    }
    
    public void compareReadPerformance() {
        System.out.println("\n=== Read Performance Comparison ===");
        
        // Traditional I/O
        Instant start = Instant.now();
        readWithTraditionalIO();
        Duration traditionalTime = Duration.between(start, Instant.now());
        
        // NIO
        start = Instant.now();
        readWithNIO();
        Duration nioTime = Duration.between(start, Instant.now());
        
        // Files utility
        start = Instant.now();
        readWithFilesUtility();
        Duration filesTime = Duration.between(start, Instant.now());
        
        System.out.println("Traditional I/O: " + traditionalTime.toMillis() + " ms");
        System.out.println("NIO: " + nioTime.toMillis() + " ms");
        System.out.println("Files utility: " + filesTime.toMillis() + " ms");
    }
    
    private void readWithTraditionalIO() {
        try (FileInputStream fis = new FileInputStream(TEST_FILE);
             BufferedInputStream bis = new BufferedInputStream(fis)) {
            
            byte[] buffer = new byte[8192];
            int totalBytes = 0;
            int bytesRead;
            
            while ((bytesRead = bis.read(buffer)) != -1) {
                totalBytes += bytesRead;
            }
            
            System.out.println("Traditional I/O read: " + totalBytes + " bytes");
            
        } catch (IOException e) {
            System.out.println("Error in traditional I/O read: " + e.getMessage());
        }
    }
    
    private void readWithNIO() {
        try (RandomAccessFile file = new RandomAccessFile(TEST_FILE, "r");
             java.nio.channels.FileChannel channel = file.getChannel()) {
            
            java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(8192);
            int totalBytes = 0;
            int bytesRead;
            
            while ((bytesRead = channel.read(buffer)) != -1) {
                totalBytes += bytesRead;
                buffer.clear();
            }
            
            System.out.println("NIO read: " + totalBytes + " bytes");
            
        } catch (IOException e) {
            System.out.println("Error in NIO read: " + e.getMessage());
        }
    }
    
    private void readWithFilesUtility() {
        try {
            byte[] data = Files.readAllBytes(Paths.get(TEST_FILE));
            System.out.println("Files utility read: " + data.length + " bytes");
        } catch (IOException e) {
            System.out.println("Error in Files utility read: " + e.getMessage());
        }
    }
    
    public void bestPractices() {
        System.out.println("\n=== I/O Best Practices ===");
        
        System.out.println("1. Use buffered streams for better performance");
        System.out.println("2. Use try-with-resources for automatic resource management");
        System.out.println("3. Choose appropriate buffer sizes (8KB is often good)");
        System.out.println("4. Use NIO for large files and high-performance requirements");
        System.out.println("5. Use Files utility methods for simple operations");
        System.out.println("6. Consider memory-mapped files for very large files");
        System.out.println("7. Use asynchronous I/O for non-blocking operations");
        System.out.println("8. Handle character encoding explicitly");
        System.out.println("9. Use appropriate stream types (byte vs character)");
        System.out.println("10. Monitor and optimize based on actual usage patterns");
    }
    
    public static void main(String[] args) {
        IOPerformanceComparison comparison = new IOPerformanceComparison();
        
        comparison.createTestFile();
        comparison.compareReadPerformance();
        comparison.bestPractices();
        
        // Cleanup
        try {
            Files.deleteIfExists(Paths.get(TEST_FILE));
        } catch (IOException e) {
            System.out.println("Error deleting test file: " + e.getMessage());
        }
    }
}
```

## Summary

Java I/O and NIO provide comprehensive capabilities for file and network operations:

### Traditional I/O (java.io)
- **Byte Streams**: For binary data (InputStream/OutputStream)
- **Character Streams**: For text data (Reader/Writer)
- **File Operations**: File class for file system operations
- **Serialization**: Object persistence mechanism
- **Buffered Streams**: Performance optimization

### New I/O (java.nio)
- **Paths and Files**: Modern file system API
- **Channels and Buffers**: High-performance I/O
- **Memory-Mapped Files**: Efficient large file handling
- **Non-blocking I/O**: Scalable network operations
- **File Watching**: Monitor file system changes
- **Asynchronous Operations**: Non-blocking file I/O

### Key Considerations
- **Performance**: NIO generally faster for large operations
- **Ease of Use**: Traditional I/O simpler for basic operations
- **Memory Usage**: NIO more memory-efficient for large files
- **Scalability**: NIO better for high-concurrency scenarios
- **Compatibility**: Traditional I/O more widely supported

### Best Practices
- Use appropriate I/O type based on requirements
- Always use try-with-resources for resource management
- Choose optimal buffer sizes
- Handle character encoding explicitly
- Consider memory-mapped files for very large files
- Use asynchronous I/O for non-blocking operations
- Monitor and optimize based on actual usage patterns
