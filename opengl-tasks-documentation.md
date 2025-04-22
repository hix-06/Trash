# OpenGL 2D Graphics Programming Tasks Documentation

This document provides a comprehensive explanation of three OpenGL graphics tasks using C++ and the GLUT library. Each task demonstrates different aspects of 2D graphics programming, from basic shape drawing to transformations and complex scene creation.

## Table of Contents
- [Task 1: House Drawing](#task-1-house-drawing)
- [Task 2: Nature Scene](#task-2-nature-scene)  
- [Task 3: Parallelogram with Transformations](#task-3-parallelogram-with-transformations)

## Task 1: House Drawing

### Objective
Create a simple 2D house using basic OpenGL primitives, with different colored components.

### Code Implementation
```cpp
#include <GL/glut.h>

void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    // House Body (Red Square)
    glColor3f(1.0f, 0.0f, 0.0f);  // Red color
    glBegin(GL_QUADS);
        glVertex2i(150, 150);  // Vertex 1
        glVertex2i(450, 150);  // Vertex 2
        glVertex2i(450, 300);  // Vertex 3
        glVertex2i(150, 300);  // Vertex 4
    glEnd();

    // Roof (Blue Triangle)
    glColor3f(0.0f, 0.0f, 1.0f);  // Blue color
    glBegin(GL_TRIANGLES);
        glVertex2i(120, 300);  // Vertex 5
        glVertex2i(480, 300);  // Vertex 6
        glVertex2i(300, 450);  // Vertex 7
    glEnd();

    // Door (Green Rectangle)
    glColor3f(0.0f, 1.0f, 0.0f);  // Green color
    glBegin(GL_QUADS);
        glVertex2i(270, 150);  // Vertex 8
        glVertex2i(330, 150);  // Vertex 9
        glVertex2i(330, 270);  // Vertex 10
        glVertex2i(270, 270);  // Vertex 11
    glEnd();

    // Left Window (White Rectangle)
    glColor3f(1.0f, 1.0f, 1.0f);  // White color
    glBegin(GL_QUADS);
        glVertex2i(180, 240);  // Vertex 12
        glVertex2i(240, 240);  // Vertex 13
        glVertex2i(240, 285);  // Vertex 14
        glVertex2i(180, 285);  // Vertex 15
    glEnd();

    // Right Window (White Rectangle)
    glBegin(GL_QUADS);
        glVertex2i(360, 240);  // Vertex 16
        glVertex2i(420, 240);  // Vertex 17
        glVertex2i(420, 285);  // Vertex 18
        glVertex2i(360, 285);  // Vertex 19
    glEnd();

    glFlush();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    
    glutInitWindowSize(600, 600);
    glutCreateWindow("House with Numbered Vertices");
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLoadIdentity();
    gluOrtho2D(0, 600, 0, 600);
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
```

### Line-by-Line Explanation

1. **Header Inclusion**:
   ```cpp
   #include <GL/glut.h>
   ```
   - Includes the GLUT (OpenGL Utility Toolkit) library for window management and drawing functions.

2. **Display Function**:
   ```cpp
   void display() {
       glClear(GL_COLOR_BUFFER_BIT);
   ```
   - `display()`: The main rendering function called by GLUT when the window needs redrawing.
   - `glClear(GL_COLOR_BUFFER_BIT)`: Clears the color buffer (screen) before drawing.

3. **House Body**:
   ```cpp
   glColor3f(1.0f, 0.0f, 0.0f);  // Red color
   glBegin(GL_QUADS);
       glVertex2i(150, 150);  // Bottom-left
       glVertex2i(450, 150);  // Bottom-right
       glVertex2i(450, 300);  // Top-right
       glVertex2i(150, 300);  // Top-left
   glEnd();
   ```
   - `glColor3f(1.0f, 0.0f, 0.0f)`: Sets the drawing color to red (RGB values).
   - `glBegin(GL_QUADS)`: Starts defining a quadrilateral (rectangle).
   - `glVertex2i(x, y)`: Specifies vertex coordinates as integers.
   - `glEnd()`: Ends the primitive definition.

4. **Roof**:
   ```cpp
   glColor3f(0.0f, 0.0f, 1.0f);  // Blue color
   glBegin(GL_TRIANGLES);
       glVertex2i(120, 300);  // Left point
       glVertex2i(480, 300);  // Right point
       glVertex2i(300, 450);  // Top point
   glEnd();
   ```
   - Sets color to blue and draws a triangle for the roof.

5. **Door**:
   ```cpp
   glColor3f(0.0f, 1.0f, 0.0f);  // Green color
   glBegin(GL_QUADS);
       glVertex2i(270, 150);  // Bottom-left
       glVertex2i(330, 150);  // Bottom-right
       glVertex2i(330, 270);  // Top-right
       glVertex2i(270, 270);  // Top-left
   glEnd();
   ```
   - Creates a green rectangle for the door.

6. **Left Window**:
   ```cpp
   glColor3f(1.0f, 1.0f, 1.0f);  // White color
   glBegin(GL_QUADS);
       glVertex2i(180, 240);  // Bottom-left
       glVertex2i(240, 240);  // Bottom-right
       glVertex2i(240, 285);  // Top-right
       glVertex2i(180, 285);  // Top-left
   glEnd();
   ```
   - Creates a white rectangle for the left window.

7. **Right Window**:
   ```cpp
   glBegin(GL_QUADS);
       glVertex2i(360, 240);  // Bottom-left
       glVertex2i(420, 240);  // Bottom-right
       glVertex2i(420, 285);  // Top-right
       glVertex2i(360, 285);  // Top-left
   glEnd();
   ```
   - Creates another white rectangle for the right window (using the previous color setting).

8. **Render**:
   ```cpp
   glFlush();
   ```
   - Forces execution of OpenGL commands, ensuring all drawing operations are processed.

9. **Main Function**:
   ```cpp
   int main(int argc, char** argv) {
       glutInit(&argc, argv);
       glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
       
       glutInitWindowSize(600, 600);
       glutCreateWindow("House with Numbered Vertices");
       glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
       glLoadIdentity();
       gluOrtho2D(0, 600, 0, 600);
       glutDisplayFunc(display);
       glutMainLoop();
       return 0;
   }
   ```
   - `glutInit(&argc, argv)`: Initializes the GLUT library.
   - `glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)`: Sets up single buffering and RGB color mode.
   - `glutInitWindowSize(600, 600)`: Sets the window size to 600×600 pixels.
   - `glutCreateWindow()`: Creates a window with the specified title.
   - `glClearColor(1.0f, 1.0f, 1.0f, 1.0f)`: Sets the background color to white.
   - `glLoadIdentity()`: Resets the current transformation matrix to identity.
   - `gluOrtho2D(0, 600, 0, 600)`: Sets up a 2D coordinate system from (0,0) to (600,600).
   - `glutDisplayFunc(display)`: Registers the display callback function.
   - `glutMainLoop()`: Enters the GLUT event processing loop.

## Task 2: Nature Scene

### Objective
Create a 2D nature scene with a sky, sun, mountains with snow peaks, grass, and trees.

### Code Implementation
```cpp
#include <GL/glut.h>
#include <cmath>

void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Sky (dark blue)
    glColor3f(0.1f, 0.1f, 0.4f);
    glBegin(GL_POLYGON);
        glVertex2i(0, 300);
        glVertex2i(800, 300);
        glVertex2i(800, 600);
        glVertex2i(0, 600);
    glEnd();

    // Grass (green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_POLYGON);
        glVertex2i(0, 0);
        glVertex2i(800, 0);
        glVertex2i(800, 300);
        glVertex2i(0, 300);
    glEnd();

    // Sun (yellow circle)
    glColor3f(1.0f, 1.0f, 0.0f);
    float x = 600, y = 450, r = 50;
    double pi = 3.141592653589793;
    glBegin(GL_TRIANGLE_FAN);
        glVertex2f(x, y);
        for(float i = 0; i <= 2 * pi; i += pi / 36) {
            glVertex2f(x + sin(i) * r, y + cos(i) * r);
        }
    glEnd();

    // Mountains (brown with white peaks)
    glColor3f(0.4f, 0.2f, 0.0f);
    for (int i = 0; i < 8; ++i) {
        int x1 = i * 100 - 50;
        int x2 = x1 + 150;
        int x3 = x1 + 75;
        glBegin(GL_TRIANGLES);
            glVertex2i(x1, 300);
            glVertex2i(x2, 300);
            glVertex2i(x3, 500);
        glEnd();
        // White peaks
        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_TRIANGLES);
            glVertex2i(x3 - 15, 460);
            glVertex2i(x3 + 15, 460);
            glVertex2i(x3, 500);
        glEnd();
        glColor3f(0.4f, 0.2f, 0.0f);
    }

    // Trees (brown trunk with green leaves)
    for (int i = 100; i < 800; i += 150) {
        // Trunk
        glColor3f(0.4f, 0.2f, 0.0f);
        glBegin(GL_POLYGON);
            glVertex2i(i - 10, 250);
            glVertex2i(i + 10, 250);
            glVertex2i(i + 10, 300);
            glVertex2i(i - 10, 300);
        glEnd();

        // Leaves (square)
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_POLYGON);
            glVertex2i(i - 25, 300);
            glVertex2i(i + 25, 300);
            glVertex2i(i + 25, 350);
            glVertex2i(i - 25, 350);
        glEnd();
    }

    glFlush();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Nature Scene");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    gluOrtho2D(0, 800, 0, 600);
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
```

### Line-by-Line Explanation

1. **Header Inclusion**:
   ```cpp
   #include <GL/glut.h>
   #include <cmath>
   ```
   - Includes GLUT library and cmath for mathematical functions (used for drawing the circular sun).

2. **Sky Drawing**:
   ```cpp
   glColor3f(0.1f, 0.1f, 0.4f);
   glBegin(GL_POLYGON);
       glVertex2i(0, 300);
       glVertex2i(800, 300);
       glVertex2i(800, 600);
       glVertex2i(0, 600);
   glEnd();
   ```
   - Creates a dark blue rectangle for the sky in the upper half of the window.

3. **Grass Drawing**:
   ```cpp
   glColor3f(0.0f, 1.0f, 0.0f);
   glBegin(GL_POLYGON);
       glVertex2i(0, 0);
       glVertex2i(800, 0);
       glVertex2i(800, 300);
       glVertex2i(0, 300);
   glEnd();
   ```
   - Creates a green rectangle for the grass in the lower half of the window.

4. **Sun Drawing**:
   ```cpp
   glColor3f(1.0f, 1.0f, 0.0f);
   float x = 600, y = 450, r = 50;
   double pi = 3.141592653589793;
   glBegin(GL_TRIANGLE_FAN);
       glVertex2f(x, y);
       for(float i = 0; i <= 2 * pi; i += pi / 36) {
           glVertex2f(x + sin(i) * r, y + cos(i) * r);
       }
   glEnd();
   ```
   - Sets color to yellow.
   - Defines the sun's center position (600,450) and radius (50).
   - Uses `GL_TRIANGLE_FAN` to draw a circle:
     - First vertex is the center point
     - Loop creates points around the circumference using sine and cosine
     - Each point is connected to the center and adjacent points

5. **Mountains Drawing**:
   ```cpp
   glColor3f(0.4f, 0.2f, 0.0f);
   for (int i = 0; i < 8; ++i) {
       int x1 = i * 100 - 50;
       int x2 = x1 + 150;
       int x3 = x1 + 75;
       glBegin(GL_TRIANGLES);
           glVertex2i(x1, 300);
           glVertex2i(x2, 300);
           glVertex2i(x3, 500);
       glEnd();
       // White peaks
       glColor3f(1.0f, 1.0f, 1.0f);
       glBegin(GL_TRIANGLES);
           glVertex2i(x3 - 15, 460);
           glVertex2i(x3 + 15, 460);
           glVertex2i(x3, 500);
       glEnd();
       glColor3f(0.4f, 0.2f, 0.0f);
   }
   ```
   - Creates 8 mountains using a loop:
     - For each mountain:
       - Calculates positions for a brown triangle
       - Draws a smaller white triangle at the peak
       - Resets color to brown for the next mountain

6. **Trees Drawing**:
   ```cpp
   for (int i = 100; i < 800; i += 150) {
       // Trunk
       glColor3f(0.4f, 0.2f, 0.0f);
       glBegin(GL_POLYGON);
           glVertex2i(i - 10, 250);
           glVertex2i(i + 10, 250);
           glVertex2i(i + 10, 300);
           glVertex2i(i - 10, 300);
       glEnd();

       // Leaves (square)
       glColor3f(0.0f, 1.0f, 0.0f);
       glBegin(GL_POLYGON);
           glVertex2i(i - 25, 300);
           glVertex2i(i + 25, 300);
           glVertex2i(i + 25, 350);
           glVertex2i(i - 25, 350);
       glEnd();
   }
   ```
   - Places trees at intervals of 150 units starting from x=100:
     - Each tree has a brown rectangle trunk
     - And a green square for leaves

7. **Main Function**:
   ```cpp
   int main(int argc, char** argv) {
       glutInit(&argc, argv);
       glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
       glutInitWindowSize(800, 600);
       glutCreateWindow("Nature Scene");
       glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
       gluOrtho2D(0, 800, 0, 600);
       glutDisplayFunc(display);
       glutMainLoop();
       return 0;
   }
   ```
   - Similar to Task 1 but with:
     - An 800×600 window size (landscape orientation)
     - A coordinate system from (0,0) to (800,600)
     - Black background color

## Task 3: Parallelogram with Transformations

### Objective
Draw shape (a parallelogram) and apply geometric transformations including rotation, scaling, and reflection.

## Code Implementation

```cpp
#include <GL/glut.h>  // Include the GLUT library
#include <cmath>      // Include for mathematical functions

// Display callback function that GLUT requires
void display() {
    glClear(GL_COLOR_BUFFER_BIT);  // Clear the screen

    // All drawing code inside the display function
    // Draw the parallelogram with transformations
    
    glPushMatrix();
    
    // Rotation Transformation
    glTranslatef(0.5f, 0.5f, 0.0f);  // Move to rotation center
    glRotatef(45.0f, 0.0f, 0.0f, 1.0f);  // Rotate 45 degrees clockwise
    glTranslatef(-0.5f, -0.5f, 0.0f);  // Move back

    // Scaling Transformation
    glScalef(1.5f, 0.8f, 1.0f);  // Scale x by 1.5 and y by 0.8

    // Reflection (X-axis)
    glScalef(1.0f, -1.0f, 1.0f);  // Flip vertically
    
    // Draw the parallelogram directly here instead of calling a function
    glBegin(GL_QUADS);  // Use quads to draw the parallelogram
    glColor3f(1.0f, 0.5f, 0.5f);  // Light red color

    // Four vertices of the parallelogram
    glVertex2f(-0.4f, -0.2f);  // Bottom-left
    glVertex2f(0.1f, -0.2f);   // Bottom-right
    glVertex2f(0.4f, 0.2f);    // Top-right
    glVertex2f(-0.1f, 0.2f);   // Top-left

    glEnd();
    
    glPopMatrix();

    glFlush();  // Render the drawing
}

// Main function
int main(int argc, char** argv) {
    glutInit(&argc, argv);  // Initialize the GLUT library
    glutCreateWindow("Parallelogram Transformations");  // Create a window
    glutInitWindowSize(600, 600);  // Set window size
    glutDisplayFunc(display);  // Set the display function
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // Background color (black)
    glutMainLoop();  // Enter the event-processing loop
    return 0;
}
```

## Code Explanation

### Header Files
```cpp
#include <GL/glut.h>  // Include the GLUT library
#include <cmath>      // Include for mathematical functions
```
- **GL/glut.h**: The OpenGL Utility Toolkit (GLUT) provides functions for window management, handling user input, and creating an OpenGL rendering context
- **cmath**: Includes standard C++ mathematical functions, although not directly used in this implementation

### Display Function
```cpp
void display() {
    glClear(GL_COLOR_BUFFER_BIT);  // Clear the screen
```
- **Purpose**: This is the main rendering function called by GLUT whenever the window needs to be redrawn
- **Return Type**: void (no return value)
- **Parameters**: None
- **glClear(GL_COLOR_BUFFER_BIT)**: Clears the color buffer (screen) to the color specified by glClearColor()

### Matrix Operations
```cpp
    glPushMatrix();
```
- **glPushMatrix()**: Saves the current transformation matrix on the stack
- This allows us to apply transformations and later restore the original state

### Rotation Transformation
```cpp
    // Rotation Transformation
    glTranslatef(0.5f, 0.5f, 0.0f);  // Move to rotation center
    glRotatef(45.0f, 0.0f, 0.0f, 1.0f);  // Rotate 45 degrees clockwise
    glTranslatef(-0.5f, -0.5f, 0.0f);  // Move back
```
- **Purpose**: Rotates the parallelogram around the point (0.5, 0.5)
- **glTranslatef(0.5f, 0.5f, 0.0f)**: Translates to the rotation center
- **glRotatef(45.0f, 0.0f, 0.0f, 1.0f)**: Rotates 45 degrees around the Z-axis (the axis perpendicular to the screen)
- **glTranslatef(-0.5f, -0.5f, 0.0f)**: Translates back to the original position
- The combination of these three operations achieves rotation around a specific point rather than the origin

### Scaling Transformation
```cpp
    // Scaling Transformation
    glScalef(1.5f, 0.8f, 1.0f);  // Scale x by 1.5 and y by 0.8
```
- **glScalef(1.5f, 0.8f, 1.0f)**: Scales the shape by factor 1.5 along the X-axis and 0.8 along the Y-axis
- This makes the parallelogram wider horizontally and shorter vertically

### Reflection Transformation
```cpp
    // Reflection (X-axis)
    glScalef(1.0f, -1.0f, 1.0f);  // Flip vertically
```
- **glScalef(1.0f, -1.0f, 1.0f)**: Reflects the shape across the X-axis by scaling the Y coordinate by -1
- This causes the parallelogram to be flipped upside down

### Drawing the Parallelogram
```cpp
    // Draw the parallelogram directly here instead of calling a function
    glBegin(GL_QUADS);  // Use quads to draw the parallelogram
    glColor3f(1.0f, 0.5f, 0.5f);  // Light red color

    // Four vertices of the parallelogram
    glVertex2f(-0.4f, -0.2f);  // Bottom-left
    glVertex2f(0.1f, -0.2f);   // Bottom-right
    glVertex2f(0.4f, 0.2f);    // Top-right
    glVertex2f(-0.1f, 0.2f);   // Top-left

    glEnd();
```
- **glBegin(GL_QUADS)**: Begins defining a quadrilateral (four-sided polygon)
- **glColor3f(1.0f, 0.5f, 0.5f)**: Sets the drawing color to light red (RGB values)
- **glVertex2f()**: Specifies the four vertices of the parallelogram using floating-point coordinates
- **glEnd()**: Ends the definition of the primitive
- The vertices are specified in counter-clockwise order, which is the standard winding order in OpenGL

### Finishing the Rendering
```cpp
    glPopMatrix();
    glFlush();  // Render the drawing
}
```
- **glPopMatrix()**: Restores the previously saved transformation matrix
- **glFlush()**: Forces execution of all OpenGL commands, ensuring all drawing operations are processed

### Main Function
```cpp
int main(int argc, char** argv) {
    glutInit(&argc, argv);  // Initialize the GLUT library
    glutCreateWindow("Parallelogram Transformations");  // Create a window
    glutInitWindowSize(600, 600);  // Set window size
    glutDisplayFunc(display);  // Set the display function
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // Background color (black)
    glutMainLoop();  // Enter the event-processing loop
    return 0;
}
```
- **Purpose**: Entry point of the program
- **Return Type**: int (standard for main functions)
- **Parameters**: 
  - **argc**: Count of command-line arguments
  - **argv**: Array of command-line argument strings
- **glutInit(&argc, argv)**: Initializes the GLUT library and processes command-line arguments
- **glutCreateWindow("Parallelogram Transformations")**: Creates a window with the specified title
- **glutInitWindowSize(600, 600)**: Sets the initial window size to 600×600 pixels
- **glutDisplayFunc(display)**: Registers the display function as the callback for rendering
- **glClearColor(0.0f, 0.0f, 0.0f, 1.0f)**: Sets the background color to black (RGBA values)
- **glutMainLoop()**: Enters the GLUT event processing loop, which handles drawing and user interaction

## Transformations Applied
1. **Rotation**: The parallelogram is rotated 45 degrees clockwise around the point (0.5, 0.5)
2. **Scaling**: The shape is stretched horizontally (1.5×) and compressed vertically (0.8×)
3. **Reflection**: The parallelogram is reflected across the X-axis

These three tasks demonstrate fundamental OpenGL concepts for 2D graphics programming, including primitive drawing, color setting, transformations, and drawing techniques for creating complex scenes.
