# 10. Polymorphism

## 📋 Overview
Polymorphism, meaning "many forms," is a fundamental principle of OOP that allows objects of different types to be treated as objects of a common base type. It enables a single interface to represent different underlying data types.

## 🎭 Types of Polymorphism

### 1. **Compile-time Polymorphism (Static)**

#### Function Overloading
```cpp
#include <iostream>
#include <string>
using namespace std;

class Calculator {
public:
    // Function overloading - same name, different parameters
    int add(int a, int b) {
        cout << "Adding two integers: ";
        return a + b;
    }
    
    double add(double a, double b) {
        cout << "Adding two doubles: ";
        return a + b;
    }
    
    int add(int a, int b, int c) {
        cout << "Adding three integers: ";
        return a + b + c;
    }
    
    string add(const string& a, const string& b) {
        cout << "Concatenating strings: ";
        return a + b;
    }
};

int main() {
    Calculator calc;
    
    cout << calc.add(5, 3) << endl;                    // Calls int version
    cout << calc.add(5.5, 3.2) << endl;               // Calls double version
    cout << calc.add(1, 2, 3) << endl;                // Calls three-parameter version
    cout << calc.add("Hello", " World") << endl;       // Calls string version
    
    return 0;
}
```

#### Operator Overloading
```cpp
#include <iostream>
using namespace std;

class Complex {
private:
    double real, imag;

public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}
    
    // Overload + operator
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    // Overload - operator
    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    // Overload << operator for output
    friend ostream& operator<<(ostream& os, const Complex& c) {
        os << c.real;
        if (c.imag >= 0) os << " + " << c.imag << "i";
        else os << " - " << (-c.imag) << "i";
        return os;
    }
    
    // Overload == operator
    bool operator==(const Complex& other) const {
        return (real == other.real) && (imag == other.imag);
    }
};

int main() {
    Complex c1(3, 4);
    Complex c2(1, 2);
    
    Complex sum = c1 + c2;      // Uses overloaded +
    Complex diff = c1 - c2;     // Uses overloaded -
    
    cout << "c1 = " << c1 << endl;
    cout << "c2 = " << c2 << endl;
    cout << "Sum = " << sum << endl;
    cout << "Difference = " << diff << endl;
    
    if (c1 == c2) {
        cout << "c1 and c2 are equal" << endl;
    } else {
        cout << "c1 and c2 are not equal" << endl;
    }
    
    return 0;
}
```

### 2. **Runtime Polymorphism (Dynamic)**

#### Virtual Functions
```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

class Animal {
protected:
    string name;

public:
    Animal(string n) : name(n) {}
    
    // Virtual function - enables runtime polymorphism
    virtual void makeSound() const {
        cout << name << " makes a generic animal sound" << endl;
    }
    
    virtual void move() const {
        cout << name << " moves in some way" << endl;
    }
    
    // Virtual destructor
    virtual ~Animal() {
        cout << "Animal " << name << " destroyed" << endl;
    }
    
    string getName() const { return name; }
};

class Dog : public Animal {
public:
    Dog(string n) : Animal(n) {}
    
    // Override virtual function
    void makeSound() const override {
        cout << name << " barks: Woof! Woof!" << endl;
    }
    
    void move() const override {
        cout << name << " runs on four legs" << endl;
    }
    
    ~Dog() {
        cout << "Dog " << name << " destroyed" << endl;
    }
};

class Cat : public Animal {
public:
    Cat(string n) : Animal(n) {}
    
    void makeSound() const override {
        cout << name << " meows: Meow! Meow!" << endl;
    }
    
    void move() const override {
        cout << name << " sneaks silently" << endl;
    }
    
    ~Cat() {
        cout << "Cat " << name << " destroyed" << endl;
    }
};

class Bird : public Animal {
public:
    Bird(string n) : Animal(n) {}
    
    void makeSound() const override {
        cout << name << " chirps: Tweet! Tweet!" << endl;
    }
    
    void move() const override {
        cout << name << " flies through the air" << endl;
    }
    
    ~Bird() {
        cout << "Bird " << name << " destroyed" << endl;
    }
};

// Function that demonstrates polymorphism
void animalActions(const Animal& animal) {
    animal.makeSound();  // Calls appropriate derived class method
    animal.move();       // Calls appropriate derived class method
}

int main() {
    // Create different animals
    Dog dog("Buddy");
    Cat cat("Whiskers");
    Bird bird("Tweety");
    
    cout << "=== Direct calls ===" << endl;
    dog.makeSound();
    cat.makeSound();
    bird.makeSound();
    
    cout << "\n=== Polymorphic calls ===" << endl;
    animalActions(dog);   // Polymorphism in action
    animalActions(cat);
    animalActions(bird);
    
    cout << "\n=== Array of base pointers ===" << endl;
    Animal* animals[] = {&dog, &cat, &bird};
    
    for (int i = 0; i < 3; i++) {
        animals[i]->makeSound();  // Runtime polymorphism
        animals[i]->move();
        cout << "---" << endl;
    }
    
    cout << "\n=== Vector of base pointers ===" << endl;
    vector<Animal*> zoo = {&dog, &cat, &bird};
    
    for (Animal* animal : zoo) {
        cout << "Animal: " << animal->getName() << endl;
        animal->makeSound();
        animal->move();
        cout << "---" << endl;
    }
    
    return 0;
}
```

## 🔧 Pure Virtual Functions and Abstract Classes

### Abstract Base Classes
```cpp
#include <iostream>
#include <vector>
#include <memory>
using namespace std;

// Abstract base class
class Shape {
protected:
    string color;

public:
    Shape(string c) : color(c) {}
    
    // Pure virtual functions make this an abstract class
    virtual double area() const = 0;
    virtual double perimeter() const = 0;
    virtual void draw() const = 0;
    
    // Regular virtual function
    virtual void displayInfo() const {
        cout << "Shape with color: " << color << endl;
        cout << "Area: " << area() << endl;
        cout << "Perimeter: " << perimeter() << endl;
    }
    
    // Virtual destructor
    virtual ~Shape() = default;
    
    string getColor() const { return color; }
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(string c, double r) : Shape(c), radius(r) {}
    
    double area() const override {
        return 3.14159 * radius * radius;
    }
    
    double perimeter() const override {
        return 2 * 3.14159 * radius;
    }
    
    void draw() const override {
        cout << "Drawing a " << color << " circle with radius " << radius << endl;
    }
};

class Rectangle : public Shape {
private:
    double length, width;

public:
    Rectangle(string c, double l, double w) : Shape(c), length(l), width(w) {}
    
    double area() const override {
        return length * width;
    }
    
    double perimeter() const override {
        return 2 * (length + width);
    }
    
    void draw() const override {
        cout << "Drawing a " << color << " rectangle " << length << "x" << width << endl;
    }
};

class Triangle : public Shape {
private:
    double side1, side2, side3;

public:
    Triangle(string c, double s1, double s2, double s3) 
        : Shape(c), side1(s1), side2(s2), side3(s3) {}
    
    double area() const override {
        // Using Heron's formula
        double s = perimeter() / 2;
        return sqrt(s * (s - side1) * (s - side2) * (s - side3));
    }
    
    double perimeter() const override {
        return side1 + side2 + side3;
    }
    
    void draw() const override {
        cout << "Drawing a " << color << " triangle with sides " 
             << side1 << ", " << side2 << ", " << side3 << endl;
    }
};

// Function to process any shape
void processShape(const Shape& shape) {
    shape.draw();
    shape.displayInfo();
    cout << "---" << endl;
}

int main() {
    // Shape shape("red");  // Error: Cannot instantiate abstract class
    
    // Create concrete shapes
    Circle circle("Red", 5.0);
    Rectangle rect("Blue", 10.0, 6.0);
    Triangle tri("Green", 3.0, 4.0, 5.0);
    
    // Process shapes polymorphically
    processShape(circle);
    processShape(rect);
    processShape(tri);
    
    // Using smart pointers for dynamic allocation
    vector<unique_ptr<Shape>> shapes;
    shapes.push_back(make_unique<Circle>("Yellow", 3.0));
    shapes.push_back(make_unique<Rectangle>("Purple", 8.0, 4.0));
    shapes.push_back(make_unique<Triangle>("Orange", 6.0, 8.0, 10.0));
    
    cout << "\n=== Processing shapes from vector ===" << endl;
    double totalArea = 0;
    for (const auto& shape : shapes) {
        shape->draw();
        totalArea += shape->area();
    }
    
    cout << "Total area of all shapes: " << totalArea << endl;
    
    return 0;
}
```

## 🎯 Virtual Function Table (vtable)

### Understanding vtable Mechanism
```cpp
#include <iostream>
using namespace std;

class Base {
public:
    virtual void func1() { cout << "Base::func1()" << endl; }
    virtual void func2() { cout << "Base::func2()" << endl; }
    void nonVirtualFunc() { cout << "Base::nonVirtualFunc()" << endl; }
    
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void func1() override { cout << "Derived::func1()" << endl; }
    // func2 not overridden - uses base version
    void nonVirtualFunc() { cout << "Derived::nonVirtualFunc()" << endl; }
};

void demonstratePolymorphism() {
    cout << "=== Direct object calls ===" << endl;
    Base baseObj;
    Derived derivedObj;
    
    baseObj.func1();        // Base::func1()
    derivedObj.func1();     // Derived::func1()
    
    cout << "\n=== Pointer-based calls ===" << endl;
    Base* ptr = &derivedObj;  // Base pointer to Derived object
    
    ptr->func1();           // Derived::func1() - virtual, uses vtable
    ptr->func2();           // Base::func2() - virtual, not overridden
    ptr->nonVirtualFunc();  // Base::nonVirtualFunc() - not virtual
    
    cout << "\n=== Reference-based calls ===" << endl;
    Base& ref = derivedObj;  // Base reference to Derived object
    
    ref.func1();            // Derived::func1() - virtual
    ref.func2();            // Base::func2() - virtual
    ref.nonVirtualFunc();   // Base::nonVirtualFunc() - not virtual
}

int main() {
    demonstratePolymorphism();
    return 0;
}
```

## 🔀 Function Overriding vs Function Hiding

### Proper Overriding
```cpp
#include <iostream>
using namespace std;

class Base {
public:
    virtual void print() const {
        cout << "Base::print()" << endl;
    }
    
    virtual void print(int x) const {
        cout << "Base::print(int): " << x << endl;
    }
    
    virtual void display() {
        cout << "Base::display()" << endl;
    }
};

class Derived : public Base {
public:
    // Proper override
    void print() const override {
        cout << "Derived::print()" << endl;
    }
    
    // This hides Base::print(int) - not polymorphic
    void print(string s) {
        cout << "Derived::print(string): " << s << endl;
    }
    
    // Proper override
    void display() override {
        cout << "Derived::display()" << endl;
    }
    
    // To make Base::print(int) available
    using Base::print;
};

int main() {
    Derived d;
    Base* ptr = &d;
    
    cout << "=== Through derived object ===" << endl;
    d.print();              // Derived::print()
    d.print(42);            // Base::print(int) - available via 'using'
    d.print("hello");       // Derived::print(string)
    
    cout << "\n=== Through base pointer ===" << endl;
    ptr->print();           // Derived::print() - polymorphic
    ptr->print(42);         // Base::print(int)
    ptr->display();         // Derived::display() - polymorphic
    
    return 0;
}
```

## 🎪 Advanced Polymorphism Concepts

### 1. **Multiple Inheritance and Virtual Functions**
```cpp
#include <iostream>
using namespace std;

class Drawable {
public:
    virtual void draw() const = 0;
    virtual ~Drawable() = default;
};

class Movable {
public:
    virtual void move(double x, double y) = 0;
    virtual ~Movable() = default;
};

class GameObject : public Drawable, public Movable {
protected:
    double posX, posY;
    string name;

public:
    GameObject(string n, double x = 0, double y = 0) 
        : name(n), posX(x), posY(y) {}
    
    void move(double x, double y) override {
        posX += x;
        posY += y;
        cout << name << " moved to (" << posX << ", " << posY << ")" << endl;
    }
    
    virtual void update() {
        cout << "Updating " << name << endl;
    }
    
    string getName() const { return name; }
};

class Player : public GameObject {
private:
    int health;

public:
    Player(string n, double x = 0, double y = 0) 
        : GameObject(n, x, y), health(100) {}
    
    void draw() const override {
        cout << "Drawing player " << name << " at (" << posX << ", " << posY 
             << ") with health " << health << endl;
    }
    
    void takeDamage(int damage) {
        health -= damage;
        cout << name << " took " << damage << " damage. Health: " << health << endl;
    }
};

class Enemy : public GameObject {
private:
    int attackPower;

public:
    Enemy(string n, int power, double x = 0, double y = 0) 
        : GameObject(n, x, y), attackPower(power) {}
    
    void draw() const override {
        cout << "Drawing enemy " << name << " at (" << posX << ", " << posY 
             << ") with attack power " << attackPower << endl;
    }
    
    void attack() {
        cout << name << " attacks with power " << attackPower << endl;
    }
};

int main() {
    Player player("Hero", 10, 20);
    Enemy enemy("Orc", 25, 50, 30);
    
    // Store in base class pointers
    vector<GameObject*> gameObjects = {&player, &enemy};
    vector<Drawable*> drawables = {&player, &enemy};
    vector<Movable*> movables = {&player, &enemy};
    
    cout << "=== Game loop simulation ===" << endl;
    for (int frame = 0; frame < 3; frame++) {
        cout << "\nFrame " << frame + 1 << ":" << endl;
        
        // Update all game objects
        for (GameObject* obj : gameObjects) {
            obj->update();
        }
        
        // Draw all drawable objects
        for (Drawable* drawable : drawables) {
            drawable->draw();
        }
        
        // Move all movable objects
        for (Movable* movable : movables) {
            movable->move(1, 1);
        }
    }
    
    return 0;
}
```

### 2. **Template Polymorphism**
```cpp
#include <iostream>
#include <vector>
#include <memory>
using namespace std;

template<typename T>
class Container {
public:
    virtual void add(const T& item) = 0;
    virtual T get(size_t index) const = 0;
    virtual size_t size() const = 0;
    virtual ~Container() = default;
};

template<typename T>
class VectorContainer : public Container<T> {
private:
    vector<T> data;

public:
    void add(const T& item) override {
        data.push_back(item);
    }
    
    T get(size_t index) const override {
        return data.at(index);
    }
    
    size_t size() const override {
        return data.size();
    }
};

template<typename T>
class ArrayContainer : public Container<T> {
private:
    T* data;
    size_t capacity;
    size_t currentSize;

public:
    ArrayContainer(size_t cap) : capacity(cap), currentSize(0) {
        data = new T[capacity];
    }
    
    ~ArrayContainer() {
        delete[] data;
    }
    
    void add(const T& item) override {
        if (currentSize < capacity) {
            data[currentSize++] = item;
        }
    }
    
    T get(size_t index) const override {
        if (index < currentSize) {
            return data[index];
        }
        throw out_of_range("Index out of range");
    }
    
    size_t size() const override {
        return currentSize;
    }
};

template<typename T>
void processContainer(Container<T>& container) {
    cout << "Container has " << container.size() << " items:" << endl;
    for (size_t i = 0; i < container.size(); i++) {
        cout << "Item " << i << ": " << container.get(i) << endl;
    }
    cout << "---" << endl;
}

int main() {
    VectorContainer<int> vecContainer;
    ArrayContainer<int> arrayContainer(5);
    
    // Add items to both containers
    for (int i = 1; i <= 3; i++) {
        vecContainer.add(i * 10);
        arrayContainer.add(i * 20);
    }
    
    // Process containers polymorphically
    processContainer(vecContainer);
    processContainer(arrayContainer);
    
    // Using with strings
    VectorContainer<string> stringContainer;
    stringContainer.add("Hello");
    stringContainer.add("World");
    stringContainer.add("Polymorphism");
    
    processContainer(stringContainer);
    
    return 0;
}
```

## 🎯 Practical Examples

### 1. **Media Player System**
```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <string>
using namespace std;

class MediaFile {
protected:
    string filename;
    double duration;  // in seconds
    string format;

public:
    MediaFile(string name, double dur, string fmt) 
        : filename(name), duration(dur), format(fmt) {}
    
    virtual ~MediaFile() = default;
    
    virtual void play() const = 0;
    virtual void pause() const = 0;
    virtual void stop() const = 0;
    virtual void displayInfo() const {
        cout << "File: " << filename << " (" << format << "), Duration: " 
             << duration << " seconds" << endl;
    }
    
    string getFilename() const { return filename; }
    double getDuration() const { return duration; }
    string getFormat() const { return format; }
};

class AudioFile : public MediaFile {
private:
    int bitrate;  // kbps
    int channels;

public:
    AudioFile(string name, double dur, int br, int ch) 
        : MediaFile(name, dur, "Audio"), bitrate(br), channels(ch) {}
    
    void play() const override {
        cout << "♪ Playing audio: " << filename << " (" << bitrate 
             << " kbps, " << channels << " channels)" << endl;
    }
    
    void pause() const override {
        cout << "⏸ Pausing audio: " << filename << endl;
    }
    
    void stop() const override {
        cout << "⏹ Stopping audio: " << filename << endl;
    }
    
    void displayInfo() const override {
        MediaFile::displayInfo();
        cout << "Bitrate: " << bitrate << " kbps, Channels: " << channels << endl;
    }
};

class VideoFile : public MediaFile {
private:
    string resolution;
    int frameRate;

public:
    VideoFile(string name, double dur, string res, int fps) 
        : MediaFile(name, dur, "Video"), resolution(res), frameRate(fps) {}
    
    void play() const override {
        cout << "▶ Playing video: " << filename << " (" << resolution 
             << ", " << frameRate << " fps)" << endl;
    }
    
    void pause() const override {
        cout << "⏸ Pausing video: " << filename << endl;
    }
    
    void stop() const override {
        cout << "⏹ Stopping video: " << filename << endl;
    }
    
    void displayInfo() const override {
        MediaFile::displayInfo();
        cout << "Resolution: " << resolution << ", Frame Rate: " << frameRate << " fps" << endl;
    }
};

class ImageFile : public MediaFile {
private:
    string resolution;
    int colorDepth;

public:
    ImageFile(string name, string res, int depth) 
        : MediaFile(name, 0, "Image"), resolution(res), colorDepth(depth) {}
    
    void play() const override {
        cout << "🖼 Displaying image: " << filename << " (" << resolution << ")" << endl;
    }
    
    void pause() const override {
        cout << "Image display paused: " << filename << endl;
    }
    
    void stop() const override {
        cout << "Closing image: " << filename << endl;
    }
    
    void displayInfo() const override {
        cout << "File: " << filename << " (" << format << "), Resolution: " 
             << resolution << ", Color Depth: " << colorDepth << " bits" << endl;
    }
};

class MediaPlayer {
private:
    vector<unique_ptr<MediaFile>> playlist;
    int currentIndex;

public:
    MediaPlayer() : currentIndex(-1) {}
    
    void addMedia(unique_ptr<MediaFile> media) {
        cout << "Added to playlist: " << media->getFilename() << endl;
        playlist.push_back(move(media));
    }
    
    void playAll() {
        cout << "\n=== Playing entire playlist ===" << endl;
        for (const auto& media : playlist) {
            media->play();
        }
    }
    
    void displayPlaylist() {
        cout << "\n=== Playlist ===" << endl;
        for (size_t i = 0; i < playlist.size(); i++) {
            cout << i + 1 << ". ";
            playlist[i]->displayInfo();
        }
    }
    
    void playMedia(size_t index) {
        if (index < playlist.size()) {
            cout << "\nPlaying media " << index + 1 << ":" << endl;
            playlist[index]->play();
        }
    }
    
    double getTotalDuration() {
        double total = 0;
        for (const auto& media : playlist) {
            total += media->getDuration();
        }
        return total;
    }
};

int main() {
    MediaPlayer player;
    
    // Add different types of media files
    player.addMedia(make_unique<AudioFile>("song1.mp3", 240, 320, 2));
    player.addMedia(make_unique<VideoFile>("movie1.mp4", 7200, "1920x1080", 30));
    player.addMedia(make_unique<ImageFile>("photo1.jpg", "4000x3000", 24));
    player.addMedia(make_unique<AudioFile>("song2.flac", 180, 1411, 2));
    player.addMedia(make_unique<VideoFile>("clip1.avi", 300, "1280x720", 24));
    
    player.displayPlaylist();
    player.playAll();
    
    cout << "\nTotal duration: " << player.getTotalDuration() << " seconds" << endl;
    
    // Play specific media
    player.playMedia(1);  // Play the video
    
    return 0;
}
```

### 2. **Drawing Application**
```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
using namespace std;

class Drawable {
public:
    virtual void draw() const = 0;
    virtual double area() const = 0;
    virtual void move(double dx, double dy) = 0;
    virtual unique_ptr<Drawable> clone() const = 0;
    virtual ~Drawable() = default;
};

class Point {
public:
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
    
    Point operator+(const Point& other) const {
        return Point(x + other.x, y + other.y);
    }
};

class Circle : public Drawable {
private:
    Point center;
    double radius;

public:
    Circle(Point c, double r) : center(c), radius(r) {}
    
    void draw() const override {
        cout << "Drawing Circle at (" << center.x << ", " << center.y 
             << ") with radius " << radius << endl;
    }
    
    double area() const override {
        return 3.14159 * radius * radius;
    }
    
    void move(double dx, double dy) override {
        center.x += dx;
        center.y += dy;
    }
    
    unique_ptr<Drawable> clone() const override {
        return make_unique<Circle>(*this);
    }
};

class Rectangle : public Drawable {
private:
    Point topLeft;
    double width, height;

public:
    Rectangle(Point tl, double w, double h) : topLeft(tl), width(w), height(h) {}
    
    void draw() const override {
        cout << "Drawing Rectangle at (" << topLeft.x << ", " << topLeft.y 
             << ") with size " << width << "x" << height << endl;
    }
    
    double area() const override {
        return width * height;
    }
    
    void move(double dx, double dy) override {
        topLeft.x += dx;
        topLeft.y += dy;
    }
    
    unique_ptr<Drawable> clone() const override {
        return make_unique<Rectangle>(*this);
    }
};

class Triangle : public Drawable {
private:
    Point p1, p2, p3;

public:
    Triangle(Point a, Point b, Point c) : p1(a), p2(b), p3(c) {}
    
    void draw() const override {
        cout << "Drawing Triangle with vertices (" << p1.x << ", " << p1.y 
             << "), (" << p2.x << ", " << p2.y << "), (" << p3.x << ", " << p3.y << ")" << endl;
    }
    
    double area() const override {
        // Using cross product formula
        return 0.5 * abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y));
    }
    
    void move(double dx, double dy) override {
        p1.x += dx; p1.y += dy;
        p2.x += dx; p2.y += dy;
        p3.x += dx; p3.y += dy;
    }
    
    unique_ptr<Drawable> clone() const override {
        return make_unique<Triangle>(*this);
    }
};

class Drawing {
private:
    vector<unique_ptr<Drawable>> shapes;

public:
    void addShape(unique_ptr<Drawable> shape) {
        shapes.push_back(move(shape));
    }
    
    void drawAll() const {
        cout << "\n=== Drawing all shapes ===" << endl;
        for (const auto& shape : shapes) {
            shape->draw();
        }
    }
    
    void moveAll(double dx, double dy) {
        cout << "\n=== Moving all shapes by (" << dx << ", " << dy << ") ===" << endl;
        for (auto& shape : shapes) {
            shape->move(dx, dy);
        }
    }
    
    double getTotalArea() const {
        double total = 0;
        for (const auto& shape : shapes) {
            total += shape->area();
        }
        return total;
    }
    
    Drawing clone() const {
        Drawing newDrawing;
        for (const auto& shape : shapes) {
            newDrawing.addShape(shape->clone());
        }
        return newDrawing;
    }
    
    void clear() {
        shapes.clear();
        cout << "Drawing cleared" << endl;
    }
    
    size_t getShapeCount() const {
        return shapes.size();
    }
};

int main() {
    Drawing drawing;
    
    // Add various shapes
    drawing.addShape(make_unique<Circle>(Point(5, 5), 3));
    drawing.addShape(make_unique<Rectangle>(Point(0, 0), 10, 8));
    drawing.addShape(make_unique<Triangle>(Point(0, 0), Point(4, 0), Point(2, 3)));
    drawing.addShape(make_unique<Circle>(Point(10, 10), 2));
    
    // Draw all shapes
    drawing.drawAll();
    
    cout << "\nTotal area: " << drawing.getTotalArea() << endl;
    cout << "Number of shapes: " << drawing.getShapeCount() << endl;
    
    // Move all shapes
    drawing.moveAll(2, 3);
    drawing.drawAll();
    
    // Clone the drawing
    cout << "\n=== Cloning drawing ===" << endl;
    Drawing clonedDrawing = drawing.clone();
    clonedDrawing.moveAll(-5, -5);
    
    cout << "\nOriginal drawing:" << endl;
    drawing.drawAll();
    
    cout << "\nCloned drawing:" << endl;
    clonedDrawing.drawAll();
    
    return 0;
}
```

## 💡 Best Practices

### 1. **Always Use Virtual Destructors**
```cpp
class Base {
public:
    virtual ~Base() = default;  // Virtual destructor
    virtual void doSomething() = 0;
};

class Derived : public Base {
private:
    int* data;
public:
    Derived() : data(new int[100]) {}
    ~Derived() override { delete[] data; }  // Will be called correctly
    void doSomething() override {}
};
```

### 2. **Use override Keyword**
```cpp
class Base {
public:
    virtual void method1() {}
    virtual void method2(int x) {}
};

class Derived : public Base {
public:
    void method1() override {}           // Good: explicit override
    void method2(double x) override {}   // Error: signature mismatch caught
};
```

### 3. **Prefer Abstract Interfaces**
```cpp
// Good: Abstract interface
class Serializable {
public:
    virtual void serialize() const = 0;
    virtual void deserialize() = 0;
    virtual ~Serializable() = default;
};

// Implement the interface
class Document : public Serializable {
public:
    void serialize() const override { /* implementation */ }
    void deserialize() override { /* implementation */ }
};
```

## 🔗 Related Topics
- [Inheritance](./09-inheritance.md)
- [Operator Overloading](./11-operator-overloading.md)
- [Templates](./12-templates.md)
- [Advanced Topics](./17-advanced-topics.md)

---
*Previous: [Inheritance](./09-inheritance.md) | Next: [Operator Overloading](./11-operator-overloading.md)*
