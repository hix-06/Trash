# OpenGL Functions Reference

This document provides a comprehensive reference for OpenGL functions used in the computer graphics lab examples. Each function is documented with its signature, purpose, and practical code examples.

## Table of Contents

1. [Core Window and Context Functions](#core-window-and-context-functions)
2. [Drawing and Primitive Functions](#drawing-and-primitive-functions)
3. [Matrix and Transformation Functions](#matrix-and-transformation-functions)
4. [Color and Material Functions](#color-and-material-functions)
5. [Viewport and Projection Functions](#viewport-and-projection-functions)
6. [Depth and Buffer Functions](#depth-and-buffer-functions)
7. [Animation and Timer Functions](#animation-and-timer-functions)

## Core Window and Context Functions

### glutInit

**Signature:**
```cpp
void glutInit(int *argc, char **argv);
```

**Purpose:**  
Initializes the GLUT library and processes command-line arguments. This must be called before any other GLUT function.

**Example:**
```cpp
int main(int argc, char** argv) {
    glutInit(&argc, argv);
    // Other initialization code
    glutMainLoop();
    return 0;
}
```

### glutCreateWindow

**Signature:**
```cpp
int glutCreateWindow(const char *title);
```

**Purpose:**  
Creates a window with the specified title for OpenGL rendering.

**Example:**
```cpp
glutCreateWindow("Red Equilateral Triangle");
```

### glutInitWindowSize

**Signature:**
```cpp
void glutInitWindowSize(int width, int height);
```

**Purpose:**  
Sets the initial window width and height in pixels.

**Example:**
```cpp
glutInitWindowSize(500, 500);
```

### glutInitWindowPosition

**Signature:**
```cpp
void glutInitWindowPosition(int x, int y);
```

**Purpose:**  
Sets the initial position of the window on the screen.

**Example:**
```cpp
glutInitWindowPosition(100, 100);
```

### glutInitDisplayMode

**Signature:**
```cpp
void glutInitDisplayMode(unsigned int mode);
```

**Purpose:**  
Sets the initial display mode for newly created windows. The mode parameter is a bitwise OR of GLUT display mode bit masks.

**Example:**
```cpp
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
```

### glutDisplayFunc

**Signature:**
```cpp
void glutDisplayFunc(void (*func)(void));
```

**Purpose:**  
Registers a callback function that will be invoked whenever GLUT determines the window needs to be redisplayed.

**Example:**
```cpp
void display() {
    // OpenGL drawing code
}

int main(int argc, char** argv) {
    // Initialization
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
```

### glutMainLoop

**Signature:**
```cpp
void glutMainLoop(void);
```

**Purpose:**  
Enters the GLUT event processing loop. This function never returns and should be called at the end of the main program.

**Example:**
```cpp
int main(int argc, char** argv) {
    // Initialization and setup
    glutMainLoop();
    return 0;  // This code is never reached
}
```

## Drawing and Primitive Functions

### glBegin / glEnd

**Signature:**
```cpp
void glBegin(GLenum mode);
void glEnd(void);
```

**Purpose:**  
Delimit the vertices of a primitive or group of primitives. All vertices specified between glBegin and glEnd are considered part of the specified primitive.

**Common modes:**
- `GL_POINTS`: Individual points
- `GL_LINES`: Line segments
- `GL_TRIANGLES`: Triangles
- `GL_QUADS`: Quadrilaterals
- `GL_POLYGON`: Simple polygon
- `GL_TRIANGLE_FAN`: Triangle fan
- `GL_QUAD_STRIP`: Quad strip

**Example (Drawing a triangle):**
```cpp
glBegin(GL_TRIANGLES);
    glVertex2f(-0.5f, -0.5f);  // First vertex
    glVertex2f(0.5f, -0.5f);   // Second vertex
    glVertex2f(0.0f, 0.5f);    // Third vertex
glEnd();
```

### glVertex2f / glVertex3f

**Signature:**
```cpp
void glVertex2f(GLfloat x, GLfloat y);
void glVertex3f(GLfloat x, GLfloat y, GLfloat z);
```

**Purpose:**  
Specify a vertex in 2D or 3D space. Must be called between glBegin and glEnd.

**Example:**
```cpp
glBegin(GL_TRIANGLES);
    glVertex2f(-0.5f, -0.5f);  // 2D point
    glVertex2f(0.5f, -0.5f);
    glVertex2f(0.0f, 0.5f);
glEnd();

// Or with 3D points
glBegin(GL_QUADS);
    glVertex3f(-1.0f, -1.0f, 1.0f);
    glVertex3f(1.0f, -1.0f, 1.0f);
    glVertex3f(1.0f, 1.0f, 1.0f);
    glVertex3f(-1.0f, 1.0f, 1.0f);
glEnd();
```

## Matrix and Transformation Functions

### glMatrixMode

**Signature:**
```cpp
void glMatrixMode(GLenum mode);
```

**Purpose:**  
Specifies which matrix stack is the target for subsequent matrix operations.

**Common modes:**
- `GL_MODELVIEW`: The modelview matrix stack
- `GL_PROJECTION`: The projection matrix stack
- `GL_TEXTURE`: The texture matrix stack

**Example:**
```cpp
glMatrixMode(GL_PROJECTION);  // Switch to projection matrix
glLoadIdentity();             // Reset the projection matrix
gluOrtho2D(-1, 1, -1, 1);     // Set up orthographic projection

glMatrixMode(GL_MODELVIEW);   // Switch back to modelview matrix
```

### glLoadIdentity

**Signature:**
```cpp
void glLoadIdentity(void);
```

**Purpose:**  
Replaces the current matrix with the identity matrix.

**Example:**
```cpp
glMatrixMode(GL_MODELVIEW);
glLoadIdentity();  // Reset the current matrix to identity
```

### glTranslatef

**Signature:**
```cpp
void glTranslatef(GLfloat x, GLfloat y, GLfloat z);
```

**Purpose:**  
Multiplies the current matrix by a translation matrix, moving objects along the x, y, and z axes.

**Example:**
```cpp
glLoadIdentity();
glTranslatef(0.0f, 0.0f, -5.0f);  // Move 5 units into the screen
```

### glRotatef

**Signature:**
```cpp
void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
```

**Purpose:**  
Multiplies the current matrix by a rotation matrix. The rotation is specified by the angle (in degrees) and an axis vector (x, y, z).

**Example:**
```cpp
glRotatef(angle, 1.0f, 1.0f, 0.0f);  // Rotate around vector (1,1,0)
```

### glPushMatrix / glPopMatrix

**Signature:**
```cpp
void glPushMatrix(void);
void glPopMatrix(void);
```

**Purpose:**  
glPushMatrix pushes the current matrix stack down, saving the current matrix. glPopMatrix pops the top matrix off the stack, restoring the previous matrix.

**Example:**
```cpp
glPushMatrix();        // Save current transformation
    glTranslatef(0.0f, 0.0f, -3.0f);
    glRotatef(angle, 1.0f, 1.0f, 0.0f);
    drawObject();      // Draw with these transformations
glPopMatrix();         // Restore original transformation
```

## Color and Material Functions

### glColor3f

**Signature:**
```cpp
void glColor3f(GLfloat red, GLfloat green, GLfloat blue);
```

**Purpose:**  
Sets the current color for drawing operations. The parameters specify the red, green, and blue components, each from 0.0 to 1.0.

**Example:**
```cpp
glColor3f(1.0f, 0.0f, 0.0f);  // Red color
glBegin(GL_TRIANGLES);
    // Vertices will be drawn in red
    glVertex2f(-0.5f, -0.5f);
    glVertex2f(0.5f, -0.5f);
    glVertex2f(0.0f, 0.5f);
glEnd();

glColor3f(0.0f, 1.0f, 0.0f);  // Green color
// Subsequent drawing will be in green
```

### glClearColor

**Signature:**
```cpp
void glClearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
```

**Purpose:**  
Specifies the clear values for the color buffers. When glClear is called with GL_COLOR_BUFFER_BIT, the color buffers are cleared to this value.

**Example:**
```cpp
glClearColor(1.0, 1.0, 1.0, 1.0);  // White background
```

## Viewport and Projection Functions

### gluOrtho2D

**Signature:**
```cpp
void gluOrtho2D(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top);
```

**Purpose:**  
Sets up a two-dimensional orthographic viewing region. This is a simplified version of glOrtho with near and far clipping planes set to -1 and 1.

**Example:**
```cpp
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluOrtho2D(-1, 1, -1, 1);  // Set coordinate system from (-1,-1) to (1,1)
```

### gluPerspective

**Signature:**
```cpp
void gluPerspective(GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar);
```

**Purpose:**  
Sets up a perspective projection matrix. Parameters are the field of view angle (in degrees) in the y direction, the aspect ratio (width divided by height), and the distances to the near and far clipping planes.

**Example:**
```cpp
glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluPerspective(45.0f, 1.0f, 0.1f, 100.0f);
```

## Depth and Buffer Functions

### glClear

**Signature:**
```cpp
void glClear(GLbitfield mask);
```

**Purpose:**  
Clears buffers to preset values. The mask is a bitwise OR of GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, and GL_STENCIL_BUFFER_BIT.

**Example:**
```cpp
// Clear both color and depth buffer
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
```

### glEnable / glDisable

**Signature:**
```cpp
void glEnable(GLenum cap);
void glDisable(GLenum cap);
```

**Purpose:**  
Enable or disable various OpenGL capabilities.

**Common capabilities:**
- `GL_DEPTH_TEST`: Enable depth testing
- `GL_LIGHTING`: Enable lighting calculations
- `GL_TEXTURE_2D`: Enable 2D texturing

**Example:**
```cpp
// Enable depth testing for 3D rendering
glEnable(GL_DEPTH_TEST);
```

### glFlush

**Signature:**
```cpp
void glFlush(void);
```

**Purpose:**  
Forces execution of OpenGL commands in finite time. All commands issued before glFlush will be executed before glFlush returns.

**Example:**
```cpp
void display() {
    // Drawing code
    glFlush();  // Ensure all drawing commands are processed
}
```

### glutSwapBuffers

**Signature:**
```cpp
void glutSwapBuffers(void);
```

**Purpose:**  
Swaps the back and front buffers for the current window if double buffering is enabled (GLUT_DOUBLE mode).

**Example:**
```cpp
void display() {
    // Drawing code
    glutSwapBuffers();  // Swap buffers to display the rendered scene
}
```

## Animation and Timer Functions

### glutTimerFunc

**Signature:**
```cpp
void glutTimerFunc(unsigned int msecs, void (*func)(int value), int value);
```

**Purpose:**  
Registers a timer callback to be triggered once after the specified number of milliseconds. The callback is given the value parameter.

**Example:**
```cpp
void update(int value) {
    // Update animation state
    glutPostRedisplay();  // Request redisplay
    glutTimerFunc(16, update, 0);  // Re-register for next frame (~60 FPS)
}

int main(int argc, char** argv) {
    // Initialization
    glutTimerFunc(0, update, 0);  // Start the timer
    glutMainLoop();
    return 0;
}
```

### glutPostRedisplay

**Signature:**
```cpp
void glutPostRedisplay(void);
```

**Purpose:**  
Marks the current window as needing to be redisplayed. At the next opportunity, the display callback will be called.

**Example:**
```cpp
void update(int value) {
    angle += 2.0f;  // Update rotation angle
    glutPostRedisplay();  // Request window redisplay
    glutTimerFunc(16, update, 0);
}
```

## Mathematical Utility Functions

### sin / cos / sqrt (from cmath library)

**Signatures:**
```cpp
float sin(float x);
float cos(float x);
float sqrt(float x);
```

**Purpose:**  
These standard C++ math functions are often used in graphics programming for calculations:
- `sin/cos`: Calculate sine and cosine of angles (in radians)
- `sqrt`: Calculate square root

**Example:**
```cpp
// Calculate point on a circle
float angle = 2.0f * M_PI * i / num_segments;
float x = radius * cos(angle);
float y = radius * sin(angle);

// Calculate height of an equilateral triangle
float side = 0.8f;
float height = side * sqrt(3) / 2;
```

## Example Shapes and Their Implementation

### Drawing an Equilateral Triangle

```cpp
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 0.0f, 0.0f);  // Red color
    
    glBegin(GL_TRIANGLES);
        float side = 0.8f;
        float height = side * sqrt(3) / 2;
        glVertex2f(-side / 2, -height / 3);  // Bottom left
        glVertex2f(side / 2, -height / 3);   // Bottom right
        glVertex2f(0.0f, 2 * height / 3);    // Top vertex
    glEnd();
    
    glFlush();
}
```

### Drawing a Circle

```cpp
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0.0f, 1.0f, 0.0f);  // Green color
    
    float cx = 0.0f;  // Center X
    float cy = 0.0f;  // Center Y
    float r = 0.3f;   // Radius
    int num_segments = 100;  // Number of segments (higher = smoother)
    
    glBegin(GL_TRIANGLE_FAN);
        glVertex2f(cx, cy);  // Center point
        for (int i = 0; i <= num_segments; ++i) {
            float theta = 2.0f * M_PI * float(i) / float(num_segments);
            float x = r * cos(theta);
            float y = r * sin(theta);
            glVertex2f(x + cx, y + cy);
        }
    glEnd();
    
    glFlush();
}
```

### Drawing a Regular Polygon (Example: Octagon)

```cpp
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0.0f, 1.0f, 0.0f);  // Green color
    
    float cx = 0.0f;  // Center X
    float cy = 0.0f;  // Center Y
    float r = 0.4f;   // Radius
    int sides = 8;    // Number of sides
    
    glBegin(GL_POLYGON);
        for (int i = 0; i < sides; ++i) {
            float angle = 2.0f * M_PI * i / sides;
            float x = cx + r * cos(angle);
            float y = cy + r * sin(angle);
            glVertex2f(x, y);
        }
    glEnd();
    
    glFlush();
}
```

### Creating a Rotating 3D Cube

```cpp
float angle = 0.0f;  // Rotation angle

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -5.0f);
    glRotatef(angle, 1.0f, 1.0f, 0.0f);  // Rotate around X and Y axes
    
    // Draw a cube with colored faces
    glBegin(GL_QUADS);
        // Front face - red
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(-1.0f, -1.0f, 1.0f);
        glVertex3f(1.0f, -1.0f, 1.0f);
        glVertex3f(1.0f, 1.0f, 1.0f);
        glVertex3f(-1.0f, 1.0f, 1.0f);
        
        // Other faces with different colors...
        // Back face - green
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(-1.0f, -1.0f, -1.0f);
        glVertex3f(-1.0f, 1.0f, -1.0f);
        glVertex3f(1.0f, 1.0f, -1.0f);
        glVertex3f(1.0f, -1.0f, -1.0f);
        
        // Top face - blue
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(-1.0f, 1.0f, -1.0f);
        glVertex3f(-1.0f, 1.0f, 1.0f);
        glVertex3f(1.0f, 1.0f, 1.0f);
        glVertex3f(1.0f, 1.0f, -1.0f);
        
        // Bottom face - yellow
        glColor3f(1.0f, 1.0f, 0.0f);
        glVertex3f(-1.0f, -1.0f, -1.0f);
        glVertex3f(1.0f, -1.0f, -1.0f);
        glVertex3f(1.0f, -1.0f, 1.0f);
        glVertex3f(-1.0f, -1.0f, 1.0f);
        
        // Right face - cyan
        glColor3f(0.0f, 1.0f, 1.0f);
        glVertex3f(1.0f, -1.0f, -1.0f);
        glVertex3f(1.0f, 1.0f, -1.0f);
        glVertex3f(1.0f, 1.0f, 1.0f);
        glVertex3f(1.0f, -1.0f, 1.0f);
        
        // Left face - magenta
        glColor3f(1.0f, 0.0f, 1.0f);
        glVertex3f(-1.0f, -1.0f, -1.0f);
        glVertex3f(-1.0f, -1.0f, 1.0f);
        glVertex3f(-1.0f, 1.0f, 1.0f);
        glVertex3f(-1.0f, 1.0f, -1.0f);
    glEnd();
    
    glutSwapBuffers();
}

void update(int value) {
    angle += 1.0f;
    if (angle > 360.0f) angle -= 360.0f;
    glutPostRedisplay();
    glutTimerFunc(16, update, 0);  // ~60 FPS
}

int main(int argc, char** argv) {
    // Initialize GLUT and create window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(600, 600);
    glutCreateWindow("Rotating Colored Cube");
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, 1.0, 1.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
    
    glutDisplayFunc(display);
    glutTimerFunc(0, update, 0);
    glutMainLoop();
    
    return 0;
}
```

### Drawing a Cylinder

```cpp
#define SLICES 50

void drawCylinder() {
    float radius = 0.5f;
    float height = 1.0f;
    
    // Draw top circle
    glBegin(GL_TRIANGLE_FAN);
        glVertex3f(0.0f, height / 2, 0.0f);  // Center point
        for(int i = 0; i <= SLICES; i++) {
            float angle = 2 * 3.14159 * i / SLICES;
            float x = cos(angle) * radius;
            float z = sin(angle) * radius;
            glVertex3f(x, height / 2, z);
        }
    glEnd();
    
    // Draw bottom circle
    glBegin(GL_TRIANGLE_FAN);
        glVertex3f(0.0f, -height / 2, 0.0f);  // Center point
        for(int i = 0; i <= SLICES; i++) {
            float angle = 2 * 3.14159 * i / SLICES;
            float x = cos(angle) * radius;
            float z = sin(angle) * radius;
            glVertex3f(x, -height / 2, z);
        }
    glEnd();
    
    // Draw side surface
    glBegin(GL_QUAD_STRIP);
        for(int i = 0; i <= SLICES; i++) {
            float angle = 2 * 3.14159 * i / SLICES;
            float x = cos(angle) * radius;
            float z = sin(angle) * radius;
            
            glVertex3f(x, height / 2, z);
            glVertex3f(x, -height / 2, z);
        }
    glEnd();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -3.0f);
    glColor3f(0.0f, 0.0f, 1.0f);  // Blue color
    drawCylinder();
    glutSwapBuffers();
}
```

## Conclusion

This reference covers the main OpenGL functions used in the textbook examples. OpenGL provides a powerful API for creating 2D and 3D graphics, from simple shapes to complex animated scenes. Understanding these core functions will provide a solid foundation for computer graphics programming with OpenGL.

For more advanced graphics programming, consider exploring modern OpenGL (3.3+) which uses shaders, vertex buffer objects, and other advanced techniques not covered in these introductory examples.
