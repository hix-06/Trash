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
Draw shape #3 from the reference image (a parallelogram) and apply geometric transformations including rotation, scaling, and reflection.

### Code Implementation (Single Main Function Approach)
```cpp
#include <GL/glut.h>
#include <cmath>

int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutCreateWindow("Parallelogram Transformations");
    glutInitWindowSize(600, 600);
    
    // Define the display callback function using a lambda
    glutDisplayFunc([]() {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);
        
        // Push matrix to save current state
        glPushMatrix();
        
        // Apply transformations
        // Translation to rotation center
        glTranslatef(0.5f, 0.5f, 0.0f);
        // Rotate 45 degrees clockwise around z-axis
        glRotatef(45.0f, 0.0f, 0.0f, 1.0f);
        // Translate back
        glTranslatef(-0.5f, -0.5f, 0.0f);
        
        // Scale: x by 1.5 and y by 0.8
        glScalef(1.5f, 0.8f, 1.0f);
        
        // Reflect across X-axis
        glScalef(1.0f, -1.0f, 1.0f);
        
        // Draw the parallelogram
        glBegin(GL_QUADS);
        glColor3f(1.0f, 0.5f, 0.5f);  // Light red color
        
        // Four vertices of parallelogram
        glVertex2f(-0.4f, -0.2f);  // Bottom-left
        glVertex2f(0.1f, -0.2f);   // Bottom-right
        glVertex2f(0.4f, 0.2f);    // Top-right
        glVertex2f(-0.1f, 0.2f);   // Top-left
        
        glEnd();
        
        // Restore the original matrix state
        glPopMatrix();
        
        // Force execution of OpenGL commands
        glFlush();
    });
    
    // Set background color to black
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    // Enter the GLUT event processing loop
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
   - Includes GLUT library and mathematical functions.

2. **Main Function and GLUT Initialization**:
   ```cpp
   int main(int argc, char** argv) {
       glutInit(&argc, argv);
       glutCreateWindow("Parallelogram Transformations");
       glutInitWindowSize(600, 600);
   ```
   - Initializes GLUT and creates a 600×600 window.

3. **Display Callback Function**:
   ```cpp
   glutDisplayFunc([]() {
       glClear(GL_COLOR_BUFFER_BIT);
   ```
   - Uses a C++ lambda function as the display callback.
   - Clears the color buffer before drawing.

4. **Matrix Operations**:
   ```cpp
   glPushMatrix();
   ```
   - Saves the current transformation matrix on the stack.

5. **Rotation Around a Point**:
   ```cpp
   // Translation to rotation center
   glTranslatef(0.5f, 0.5f, 0.0f);
   // Rotate 45 degrees clockwise around z-axis
   glRotatef(45.0f, 0.0f, 0.0f, 1.0f);
   // Translate back
   glTranslatef(-0.5f, -0.5f, 0.0f);
   ```
   - This sequence performs rotation around point (0.5, 0.5):
     1. First translates to the rotation center
     2. Applies a 45-degree rotation around the Z-axis
     3. Translates back to preserve position

6. **Scaling**:
   ```cpp
   // Scale: x by 1.5 and y by 0.8
   glScalef(1.5f, 0.8f, 1.0f);
   ```
   - Stretches the shape horizontally by 1.5× and compresses it vertically by 0.8×.

7. **Reflection**:
   ```cpp
   // Reflect across X-axis
   glScalef(1.0f, -1.0f, 1.0f);
   ```
   - Flips the shape vertically by scaling the y-coordinate by -1.

8. **Drawing the Parallelogram**:
   ```cpp
   glBegin(GL_QUADS);
   glColor3f(1.0f, 0.5f, 0.5f);  // Light red color
   
   // Four vertices of parallelogram
   glVertex2f(-0.4f, -0.2f);  // Bottom-left
   glVertex2f(0.1f, -0.2f);   // Bottom-right
   glVertex2f(0.4f, 0.2f);    // Top-right
   glVertex2f(-0.1f, 0.2f);   // Top-left
   
   glEnd();
   ```
   - Sets color to light red.
   - Defines the parallelogram using four vertices in GL_QUADS mode.

9. **Restoring Matrix State**:
   ```cpp
   glPopMatrix();
   ```
   - Restores the previously saved transformation matrix.

10. **Rendering and Main Loop**:
    ```cpp
    glFlush();
    });
    
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glutMainLoop();
    
    return 0;
    ```
    - `glFlush()`: Forces execution of OpenGL commands.
    - `glClearColor()`: Sets the background color to black.
    - `glutMainLoop()`: Enters the event processing loop.

## Summary of OpenGL Functions Used

1. **Window Management**:
   - `glutInit(&argc, argv)`: Initializes GLUT library
   - `glutInitDisplayMode()`: Sets display mode (buffering and color model)
   - `glutInitWindowSize()`: Sets window dimensions
   - `glutCreateWindow()`: Creates a named window
   - `glutMainLoop()`: Enters the event processing loop

2. **Coordinate System Setup**:
   - `gluOrtho2D()`: Sets up a 2D coordinate system
   - `glLoadIdentity()`: Resets the current matrix to identity

3. **Drawing Functions**:
   - `glBegin()/glEnd()`: Delimits vertex specifications for primitives
   - `glVertex2i()/glVertex2f()`: Specifies vertex coordinates (integer/float)
   - `glColor3f()`: Sets the current drawing color
   - `glClear()`: Clears buffers to preset values
   - `glClearColor()`: Specifies clear values for the color buffer
   - `glFlush()`: Forces execution of OpenGL commands

4. **Transformation Functions**:
   - `glPushMatrix()/glPopMatrix()`: Saves/restores the current matrix
   - `glTranslatef()`: Multiplies the current matrix by a translation matrix
   - `glRotatef()`: Multiplies the current matrix by a rotation matrix
   - `glScalef()`: Multiplies the current matrix by a scaling matrix

5. **Primitive Types**:
   - `GL_QUADS`: Four-vertex quadrilaterals
   - `GL_TRIANGLES`: Three-vertex triangles
   - `GL_POLYGON`: Multiple-vertex convex polygons
   - `GL_TRIANGLE_FAN`: Triangle fan primitive (first vertex is center)

These three tasks demonstrate fundamental OpenGL concepts for 2D graphics programming, including primitive drawing, color setting, transformations, and drawing techniques for creating complex scenes.
