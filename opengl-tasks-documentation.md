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
Create a visualization of various 2D transformations (translation, scaling, rotation, and reflection) applied to a parallelogram using OpenGL.

### Code Implementation
```cpp
#include <GL/glut.h>
#include <cmath>

// Function to draw a translated parallelogram
void translate_shape(float tx, float ty) {
    glColor3f(1.0f, 0.5f, 0.5f); // Light red
    glBegin(GL_QUADS);
    glVertex2f(260 + tx, 290 + ty);
    glVertex2f(270 + tx, 330 + ty);
    glVertex2f(320 + tx, 330 + ty);
    glVertex2f(310 + tx, 290 + ty);
    glEnd();
}

// Function to draw a scaled parallelogram
void scale_shape(float sx, float sy) {
    glColor3f(0.5f, 1.0f, 0.5f); // Light green
    glBegin(GL_QUADS);
    float cx = 290, cy = 310;
    glVertex2f((260 - cx) * sx + cx, (290 - cy) * sy + cy);
    glVertex2f((270 - cx) * sx + cx, (330 - cy) * sy + cy);
    glVertex2f((320 - cx) * sx + cx, (330 - cy) * sy + cy);
    glVertex2f((310 - cx) * sx + cx, (290 - cy) * sy + cy);
    glEnd();
}

// Function to rotate the parallelogram
void rotate_shape(float angle) {
    glColor3f(0.5f, 0.5f, 1.0f); // Light blue
    glBegin(GL_QUADS);
    float cx = 290, cy = 310;
    float rad = angle * 3.14159f / 180.0f;
    float cosA = cos(rad);
    float sinA = sin(rad);
    glVertex2f(cosA * (260 - cx) - sinA * (290 - cy) + cx, 
               sinA * (260 - cx) + cosA * (290 - cy) + cy);
    glVertex2f(cosA * (270 - cx) - sinA * (330 - cy) + cx, 
               sinA * (270 - cx) + cosA * (330 - cy) + cy);
    glVertex2f(cosA * (320 - cx) - sinA * (330 - cy) + cx, 
               sinA * (320 - cx) + cosA * (330 - cy) + cy);
    glVertex2f(cosA * (310 - cx) - sinA * (290 - cy) + cx, 
               sinA * (310 - cx) + cosA * (290 - cy) + cy);
    glEnd();
}

// Function to reflect the parallelogram over x-axis
void reflect_shape_x() {
    glColor3f(1.0f, 1.0f, 0.5f); // Light yellow
    glBegin(GL_QUADS);
    float cy = 310; // Reflection axis is y=cy
    glVertex2f(260, 2*cy - 290);
    glVertex2f(270, 2*cy - 330);
    glVertex2f(320, 2*cy - 330);
    glVertex2f(310, 2*cy - 290);
    glEnd();
}

// Function to reflect the parallelogram over y-axis
void reflect_shape_y() {
    glColor3f(1.0f, 1.0f, 0.5f); // Light yellow
    glBegin(GL_QUADS);
    float cx = 290; // Reflection axis is x=cx
    glVertex2f(2*cx - 260, 290);
    glVertex2f(2*cx - 270, 330);
    glVertex2f(2*cx - 320, 330);
    glVertex2f(2*cx - 310, 290);
    glEnd();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Original shape (center)
    glColor3f(1.0f, 1.0f, 1.0f); // White
    glBegin(GL_QUADS);
    glVertex2f(260, 290);
    glVertex2f(270, 330);
    glVertex2f(320, 330);
    glVertex2f(310, 290);
    glEnd();
    
    // Translated (bottom-left)
    translate_shape(-150, -150);
    
    // Scaled (top-right)
    glPushMatrix();
    glTranslatef(150, 150, 0);
    scale_shape(1.5f, 1.5f);
    glPopMatrix();
    
    // Rotated (top-left)
    glPushMatrix();
    glTranslatef(-150, 150, 0);
    rotate_shape(45);
    glPopMatrix();
    
    // Reflected over y-axis (bottom-right)
    glPushMatrix();
    glTranslatef(150, -150, 0);
    reflect_shape_y();
    glPopMatrix();
    
    glFlush();
}

void init() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, 600.0, 0.0, 600.0);
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitWindowSize(600, 600);
    glutCreateWindow("Parallelogram Transformations");
    init();
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
   - Includes the GLUT library for OpenGL functionality and the cmath library for mathematical functions (sin, cos).

2. **Translation Function**:
   ```cpp
   void translate_shape(float tx, float ty) {
       glColor3f(1.0f, 0.5f, 0.5f); // Light red
       glBegin(GL_QUADS);
       glVertex2f(260 + tx, 290 + ty);
       glVertex2f(270 + tx, 330 + ty);
       glVertex2f(320 + tx, 330 + ty);
       glVertex2f(310 + tx, 290 + ty);
       glEnd();
   }
   ```
   - Sets the color to light red.
   - Creates a parallelogram by adding the translation values (`tx`, `ty`) to each vertex coordinate.
   - Translation is a simple addition of offset values to the original coordinates.

3. **Scaling Function**:
   ```cpp
   void scale_shape(float sx, float sy) {
       glColor3f(0.5f, 1.0f, 0.5f); // Light green
       glBegin(GL_QUADS);
       float cx = 290, cy = 310;
       glVertex2f((260 - cx) * sx + cx, (290 - cy) * sy + cy);
       glVertex2f((270 - cx) * sx + cx, (330 - cy) * sy + cy);
       glVertex2f((320 - cx) * sx + cx, (330 - cy) * sy + cy);
       glVertex2f((310 - cx) * sx + cx, (290 - cy) * sy + cy);
       glEnd();
   }
   ```
   - Sets the color to light green.
   - Defines a center point (`cx`, `cy`) around which scaling occurs.
   - For each vertex:
     1. Translates the point relative to the center point
     2. Applies the scaling factors (`sx`, `sy`)
     3. Translates back to the original coordinate system
   - This ensures scaling happens around the center point rather than the origin.

4. **Rotation Function**:
   ```cpp
   void rotate_shape(float angle) {
       glColor3f(0.5f, 0.5f, 1.0f); // Light blue
       glBegin(GL_QUADS);
       float cx = 290, cy = 310;
       float rad = angle * 3.14159f / 180.0f;
       float cosA = cos(rad);
       float sinA = sin(rad);
       glVertex2f(cosA * (260 - cx) - sinA * (290 - cy) + cx, 
                  sinA * (260 - cx) + cosA * (290 - cy) + cy);
       glVertex2f(cosA * (270 - cx) - sinA * (330 - cy) + cx, 
                  sinA * (270 - cx) + cosA * (330 - cy) + cy);
       glVertex2f(cosA * (320 - cx) - sinA * (330 - cy) + cx, 
                  sinA * (320 - cx) + cosA * (330 - cy) + cy);
       glVertex2f(cosA * (310 - cx) - sinA * (290 - cy) + cx, 
                  sinA * (310 - cx) + cosA * (290 - cy) + cy);
       glEnd();
   }
   ```
   - Sets the color to light blue.
   - Converts the angle from degrees to radians.
   - Pre-computes sine and cosine values for efficiency.
   - For each vertex:
     1. Translates the point relative to the center point (`cx`, `cy`)
     2. Applies the 2D rotation matrix: [x' = x*cos(θ) - y*sin(θ), y' = x*sin(θ) + y*cos(θ)]
     3. Translates back to the original coordinate system
   - This implements rotation around a specific center point rather than the origin.

5. **X-Axis Reflection Function**:
   ```cpp
   void reflect_shape_x() {
       glColor3f(1.0f, 1.0f, 0.5f); // Light yellow
       glBegin(GL_QUADS);
       float cy = 310; // Reflection axis is y=cy
       glVertex2f(260, 2*cy - 290);
       glVertex2f(270, 2*cy - 330);
       glVertex2f(320, 2*cy - 330);
       glVertex2f(310, 2*cy - 290);
       glEnd();
   }
   ```
   - Sets the color to light yellow.
   - Reflects the parallelogram about the horizontal line y=310 (the center's y-coordinate).
   - For each y-coordinate, the new position is calculated as: y' = 2*cy - y
   - The x-coordinates remain unchanged during reflection across a horizontal axis.

6. **Y-Axis Reflection Function**:
   ```cpp
   void reflect_shape_y() {
       glColor3f(1.0f, 1.0f, 0.5f); // Light yellow
       glBegin(GL_QUADS);
       float cx = 290; // Reflection axis is x=cx
       glVertex2f(2*cx - 260, 290);
       glVertex2f(2*cx - 270, 330);
       glVertex2f(2*cx - 320, 330);
       glVertex2f(2*cx - 310, 290);
       glEnd();
   }
   ```
   - Sets the color to light yellow.
   - Reflects the parallelogram about the vertical line x=290 (the center's x-coordinate).
   - For each x-coordinate, the new position is calculated as: x' = 2*cx - x
   - The y-coordinates remain unchanged during reflection across a vertical axis.

7. **Display Function**:
   ```cpp
   void display() {
       glClear(GL_COLOR_BUFFER_BIT);
       
       // Original shape (center)
       glColor3f(1.0f, 1.0f, 1.0f); // White
       glBegin(GL_QUADS);
       glVertex2f(260, 290);
       glVertex2f(270, 330);
       glVertex2f(320, 330);
       glVertex2f(310, 290);
       glEnd();
       
       // Translated (bottom-left)
       translate_shape(-150, -150);
       
       // Scaled (top-right)
       glPushMatrix();
       glTranslatef(150, 150, 0);
       scale_shape(1.5f, 1.5f);
       glPopMatrix();
       
       // Rotated (top-left)
       glPushMatrix();
       glTranslatef(-150, 150, 0);
       rotate_shape(45);
       glPopMatrix();
       
       // Reflected over y-axis (bottom-right)
       glPushMatrix();
       glTranslatef(150, -150, 0);
       reflect_shape_y();
       glPopMatrix();
       
       glFlush();
   }
   ```
   - Clears the color buffer before drawing.
   - Draws the original white parallelogram in the center.
   - Draws a translated version in the bottom-left quadrant (moved by -150 units in both x and y).
   - Uses `glPushMatrix()` and `glPopMatrix()` to isolate transformations:
     - For the scaled version, translates to the top-right quadrant before scaling by 1.5x.
     - For the rotated version, translates to the top-left quadrant before rotating by 45 degrees.
     - For the reflected version, translates to the bottom-right quadrant before reflecting over the y-axis.
   - `glFlush()` ensures all drawing commands are executed.

8. **Initialization Function**:
   ```cpp
   void init() {
       glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
       glMatrixMode(GL_PROJECTION);
       gluOrtho2D(0.0, 600.0, 0.0, 600.0);
   }
   ```
   - Sets the background color to black.
   - Sets up a 2D orthographic projection with coordinates ranging from (0,0) to (600,600).

9. **Main Function**:
   ```cpp
   int main(int argc, char** argv) {
       glutInit(&argc, argv);
       glutInitWindowSize(600, 600);
       glutCreateWindow("Parallelogram Transformations");
       init();
       glutDisplayFunc(display);
       glutMainLoop();
       return 0;
   }
   ```
   - Initializes the GLUT library.
   - Creates a 600×600 pixel window titled "Parallelogram Transformations."
   - Calls the `init()` function to set up the OpenGL environment.
   - Registers the `display()` function as the callback for rendering.
   - Enters the GLUT event processing loop.

### Transformation Concepts Illustrated

1. **Translation**: Moving an object by adding offset values to coordinates.
   - Implementation: Adding (`tx`, `ty`) to each vertex.
   - Example: The red parallelogram is translated by (-150, -150).

2. **Scaling**: Changing the size of an object.
   - Implementation: Multiplying coordinates by scale factors relative to a center point.
   - Example: The green parallelogram is scaled by 1.5 times in both dimensions.

3. **Rotation**: Turning an object around a center point.
   - Implementation: Applying a rotation matrix to vertex coordinates.
   - Example: The blue parallelogram is rotated by 45 degrees.

4. **Reflection**: Mirroring an object across an axis.
   - Implementation: Reflecting coordinates across a line (y=cy or x=cx).
   - Example: The yellow parallelogram is reflected across the y-axis.

5. **Matrix Operations**: Using OpenGL's matrix stack (`glPushMatrix()` and `glPopMatrix()`) to isolate transformations.
   - This prevents transformations from affecting other objects in the scene.
