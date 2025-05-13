# OpenGL Shape Drawing Reference Guide

This documentation organizes various OpenGL shape drawing techniques into related groups to make learning and memorization easier. By understanding the relationships between shapes, you can more easily recall how to draw any shape needed for your interview.

## Base Framework

```cpp
#include <iostream>
#include <GL/glut.h>
#include <cmath>

void display() 
{
    glClear(GL_COLOR_BUFFER_BIT);
    
    glColor3f(1.0f, 0.0f, 0.0f);  // Red color
    glBegin(GL_POLYGON);  // GL_TRIANGLES, GL_QUADS, GL_POLYGON or GL_TRIANGLE_FAN
    
    // drawing logic and coordinates with glVertex2f()
    float cx = 0.3f;  // Center x
    float cy = 0.6f;  // Center y
    float s = 0.2f;   // Size/scale factor
    
    // Shape drawing code will go here
    
    glEnd();
    
    glFlush();
}

void init()
{
    glClearColor(1.0, 1.0, 1.0, 1.0);  // White background
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);  // Coordinate system: (-1,-1) to (1,1)
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutCreateWindow(" ");
    glutInitWindowSize(500, 500);
    glutDisplayFunc(display);
    init();
    glutMainLoop();
}
```

**Note:** All coordinates in `glVertex2f(x, y)` are measured from the center point (0, 0). `gluOrtho2D(-1, 1, -1, 1)` creates a coordinate system where (0, 0) is the center, (-1, -1) is bottom-left, and (1, 1) is top-right.

## 1. Basic Polygons

### Group 1: Triangles and Variations

#### Standard Triangle (Shape 1)
```cpp
// Triangle pointing up
glVertex2f(cx,     cy + s);      // Top
glVertex2f(cx - s, cy - s);      // Bottom-left
glVertex2f(cx + s, cy - s);      // Bottom-right
```

#### Right Triangle (Shape 2)
```cpp
// Triangle with 90° angle
glVertex2f(cx - s, cy + s);   // Top-left
glVertex2f(cx - s, cy - s);   // Bottom-left (right angle)
glVertex2f(cx + s, cy - s);   // Bottom-right
```

### Group 2: Quadrilaterals

#### Parallelogram (Shape 3)
```cpp
// Slanted Parallelogram
glVertex2f(cx - s, cy - s);   // Bottom-left
glVertex2f(cx,     cy - s);   // Bottom-right
glVertex2f(cx + s, cy + s);   // Top-right
glVertex2f(cx,     cy + s);   // Top-left
```

#### Rotated Square (Shape 4)
```cpp
// Diamond/Rotated Square
glVertex2f(cx,     cy + s);   // Top
glVertex2f(cx + s, cy);       // Right
glVertex2f(cx,     cy - s);   // Bottom
glVertex2f(cx - s, cy);       // Left
```

#### Trapezoid (Shape 15)
```cpp
// Trapezoid (narrower at top)
glVertex2f(cx - s,     cy - s);       // Bottom-left
glVertex2f(cx + s,     cy - s);       // Bottom-right
glVertex2f(cx + s*0.65, cy + s*0.65); // Top-right
glVertex2f(cx - s*0.65, cy + s*0.65); // Top-left
```

## 2. Regular Polygons Using Loops

### Group 1: Regular Polygons

#### Hexagon (Shape 9)
```cpp
// Regular hexagon (6 sides)
for (int i = 0; i < 6; i++) {
    float angle = i * 2 * M_PI / 6;
    glVertex2f(cx + s * cos(angle), cy + s * sin(angle));
}
```

#### Octagon (Shape 6)
```cpp
// Regular octagon (8 sides)
for (int i = 0; i < 8; i++) {
    float angle = i * 2 * M_PI / 8;
    glVertex2f(cx + s * cos(angle), cy + s * sin(angle));
}
```

### Group 2: Curved Shapes

#### Circle (Full 360°)
```cpp
// Circle
for (int i = 0; i < 360; i++) {
    float angle = i * M_PI / 180.0f;
    glVertex2f(cx + s * cos(angle), cy + s * sin(angle));
}
```

#### Oval/Ellipse (Shape 5)
```cpp
// Oval (stretched horizontally)
for (int i = 0; i < 360; i++) {
    float angle = i * M_PI / 180.0f;
    glVertex2f(cx + s * 1.3f * cos(angle), cy + s * 0.7f * sin(angle));
}
```

#### Half Circle (Shape 18)
```cpp
// Half circle (downward facing)
for (int i = 0; i < 180; i++) {
    float angle = i * M_PI / 180.0f;
    glVertex2f(cx + s * cos(-angle), cy + s * sin(-angle));
}
```

## 3. Stars and Special Shapes

### Group 1: Stars (Using GL_TRIANGLE_FAN)

#### Chubby 5-Point Star (Shape 8)
```cpp
// GL_TRIANGLE_FAN mode
glVertex2f(cx, cy); // center point
for (int i = 0; i <= 10; i++) {
    float angle = i * M_PI / 5 - M_PI/3.4;
    float radius = (i % 2 == 0) ? s : s / 2; // 2: thickness ratio
    glVertex2f(cx + radius * cos(angle), cy + radius * sin(angle));
}
```

#### Narrow 5-Point Star (Shape 13)
```cpp
// GL_TRIANGLE_FAN mode
glVertex2f(cx, cy); // center point
for (int i = 0; i <= 10; i++) {
    float angle = i * M_PI / 5 - M_PI/3.4;
    float radius = (i % 2 == 0) ? s : s / 2.7; // 2.7: thickness ratio (narrower)
    glVertex2f(cx + radius * cos(angle), cy + radius * sin(angle));
}
```

### Group 2: Composite Shapes (Multiple glBegin/End calls)

#### Plus Sign (Shape 14)
```cpp
// Vertical bar
glColor3f(1.0f, 0.0f, 0.0f);  // Red
glBegin(GL_QUADS);
glVertex2f(cx - s * 0.15f, cy + s * 0.6f);
glVertex2f(cx + s * 0.15f, cy + s * 0.6f);
glVertex2f(cx + s * 0.15f, cy - s * 0.6f);
glVertex2f(cx - s * 0.15f, cy - s * 0.6f);
glEnd();

// Horizontal bar
glColor3f(0.0f, 1.0f, 0.0f);  // Green
glBegin(GL_QUADS);
glVertex2f(cx - s * 0.6f, cy + s * 0.15f);
glVertex2f(cx + s * 0.6f, cy + s * 0.15f);
glVertex2f(cx + s * 0.6f, cy - s * 0.15f);
glVertex2f(cx - s * 0.6f, cy - s * 0.15f);
glEnd();
```

#### Circle with Plus (Shape 10)
```cpp
// Circle
glBegin(GL_POLYGON);
for (int i = 0; i < 360; i++) {
    float angle = i * M_PI / 180.0f;
    glVertex2f(cx + s * cos(angle), cy + s * sin(angle));
}
glEnd();

// Plus (vertical bar)
glColor3f(1.0f, 1.0f, 1.0f);  // White
glBegin(GL_QUADS);
glVertex2f(cx - s * 0.15f, cy + s * 0.6f);
glVertex2f(cx + s * 0.15f, cy + s * 0.6f);
glVertex2f(cx + s * 0.15f, cy - s * 0.6f);
glVertex2f(cx - s * 0.15f, cy - s * 0.6f);
glEnd();

// Plus (horizontal bar)
glColor3f(1.0f, 1.0f, 1.0f);  // White
glBegin(GL_QUADS);
glVertex2f(cx - s * 0.6f, cy + s * 0.15f);
glVertex2f(cx + s * 0.6f, cy + s * 0.15f);
glVertex2f(cx + s * 0.6f, cy - s * 0.15f);
glVertex2f(cx - s * 0.6f, cy - s * 0.15f);
glEnd();
```

#### Arrow (Shape 11)
```cpp
// Triangle head
glBegin(GL_TRIANGLES);
glVertex2f(cx + s, cy);       // Tip
glVertex2f(cx,     cy + s);   // Top
glVertex2f(cx,     cy - s);   // Bottom
glEnd();

// Rectangle tail
glBegin(GL_QUADS);
glVertex2f(cx - s,     cy + s * 0.5f);
glVertex2f(cx,         cy + s * 0.5f);
glVertex2f(cx,         cy - s * 0.5f);
glVertex2f(cx - s,     cy - s * 0.5f);
glEnd();
```

## 4. 3D-like Shapes

### Group 1: Simple 3D Objects

#### Box 3D (Shape 20)
```cpp
// Front face
glColor3f(1.0f, 0.0f, 0.0f);  // Red
glBegin(GL_QUADS);
glVertex2f(cx - s,     cy - s);
glVertex2f(cx,         cy - s);
glVertex2f(cx,         cy);
glVertex2f(cx - s,     cy);
glEnd();

// Top face
glColor3f(0.0f, 1.0f, 0.0f);  // Green
glBegin(GL_QUADS);
glVertex2f(cx - s,     cy);
glVertex2f(cx,         cy);
glVertex2f(cx + s*0.3, cy + s*0.3);
glVertex2f(cx - s*0.7, cy + s*0.3);
glEnd();

// Side face
glColor3f(0.0f, 0.0f, 1.0f);  // Blue
glBegin(GL_QUADS);
glVertex2f(cx,         cy - s);
glVertex2f(cx + s*0.3, cy - s + s*0.3);
glVertex2f(cx + s*0.3, cy + s*0.3);
glVertex2f(cx,         cy);
glEnd();
```

#### 2.5D Box (Shape 7)
```cpp
// Top face
glColor3f(0.6f, 0.6f, 0.6f); // gray
glBegin(GL_POLYGON);
glVertex2f(cx, cy);
glVertex2f(cx + s*.7f, cy + s*.5f);
glVertex2f(cx, cy + 2 * s*.5f);
glVertex2f(cx - s*.7f, cy + s*.5f);
glEnd();

// Right face
glColor3f(0.6f, 1.0f, 0.6f); // greenish
glBegin(GL_POLYGON);
glVertex2f(cx, cy);
glVertex2f(cx + s*.7f, cy + s*.5f);
glVertex2f(cx + s*.7f, cy - s + s*.5f);
glVertex2f(cx, cy - s);
glEnd();

// Left face
glColor3f(1.0f, 0.4f, 0.6f); // pink
glBegin(GL_POLYGON);
glVertex2f(cx, cy);
glVertex2f(cx - s*.7f, cy + s*.5f);
glVertex2f(cx - s*.7f, cy - s + s*.5f);
glVertex2f(cx, cy - s);
glEnd();
```

## Memory Tips

1. **Basic Structure**: Remember that most shapes use:
   - A center point (cx, cy)
   - A size parameter (s)
   - glVertex2f() calls to define vertices

2. **Pattern Recognition**:
   - Regular polygons: Always use a loop with angle = i * 2 * M_PI / n (where n is number of sides)
   - Stars: Use GL_TRIANGLE_FAN with alternating radii
   - 3D shapes: Draw each face separately with different colors

3. **Formulaic Similarities**:
   - For any circular shape: x = cx + radius * cos(angle), y = cy + radius * sin(angle)
   - For regular polygons: Just change the number of sides in the loop
   - For 3D effects: Add an offset to coordinates (typically 0.3*s or 0.5*s)

4. **Coordinate System Reminder**:
   - (0,0) is the center of the window
   - +x goes right, +y goes up
   - Values range from -1 to 1 in both directions

## OpenGL Shape Transformations: Translation, Scaling, and Reflection

This code demonstrates basic OpenGL transformations by drawing a circle in each quadrant of the screen with different transformations applied.

```cpp
#include <iostream>
#include <GL/glut.h>
#include <cmath>

void drawCircle(float cx, float cy, float s) {
    glBegin(GL_POLYGON);  
    for (int i = 0; i < 360; i++) {
        float angle = i * M_PI / 180.0f;
        glVertex2f(cx + s * cos(angle), cy + s * sin(angle));
    }
    glEnd();
}

void display() 
{
    glClear(GL_COLOR_BUFFER_BIT);
    
    // 1. Original shape - top right quadrant
    glColor3f(1.0f, 0.0f, 0.0f);  // Red color
    glPushMatrix();  // Save the current matrix
    drawCircle(0.3f, 0.6f, 0.2f);
    glPopMatrix();  // Restore the matrix
    
    // 2. Translated shape - top left quadrant
    glColor3f(0.0f, 1.0f, 0.0f);  // Green color
    glPushMatrix();
    glTranslatef(-0.6f, 0.0f, 0.0f);  // Move to the left
    drawCircle(0.3f, 0.6f, 0.2f);
    glPopMatrix();
    
    // 3. Scaled shape - bottom right quadrant
    glColor3f(0.0f, 0.0f, 1.0f);  // Blue color
    glPushMatrix();
    glTranslatef(0.0f, -1.2f, 0.0f);  // Move down
    glScalef(1.5f, 1.5f, 1.0f);  // Scale by 1.5 times
    drawCircle(0.3f, 0.6f, 0.2f);
    glPopMatrix();
    
    // 4. Reflected shape - bottom left quadrant
    glColor3f(1.0f, 1.0f, 0.0f);  // Yellow color
    glPushMatrix();
    // Reflect across y-axis (multiply x by -1) and move down
    glScalef(-1.0f, 1.0f, 1.0f);  // Reflection about y-axis
    glTranslatef(0.0f, -1.2f, 0.0f);  // Move down
    drawCircle(0.3f, 0.6f, 0.2f);
    glPopMatrix();
    
    glFlush();
}

void init()
{
    glClearColor(1.0, 1.0, 1.0, 1.0);  // White background
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);  // Coordinate system: (-1,-1) to (1,1)
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitWindowSize(500, 500);
    glutCreateWindow("OpenGL Transformations");
    glutDisplayFunc(display);
    init();
    glutMainLoop();
    
    return 0;
}
```

### Transformation Methods Used

1. **glTranslatef(x, y, z)** - Moves objects in coordinate space
2. **glScalef(x, y, z)** - Resizes objects along each axis
3. **glScalef** with negative values - Creates reflections across axes

The program displays:
- Red circle (original) in top-right quadrant
- Green circle (translated) in top-left quadrant
- Blue circle (scaled) in bottom-right quadrant
- Yellow circle (reflected) in bottom-left quadrant

## Table of Contents
1. [Mouse Control Example](#mouse-control-example)
2. [Animation Example](#animation-example)
3. [Keyboard Control Example](#keyboard-control-example)

## Mouse Control Example

```cpp
#include <iostream>
#include <GL/glut.h>
#include <cmath>

// Circle properties
float cx = 0.0f, cy = 0.0f;    // Center position
float s = 0.2f;                // Size/scale

// Animation properties
float dx = 0.01f, dy = 0.01f;  // Velocity
bool animationEnabled = false; // Animation starts paused

void display() {
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Draw red circle
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POLYGON);
    for (int i = 0; i < 360; i++) {
        float angle = i * M_PI / 180.0f;
        glVertex2f(cx + s * cos(angle), cy + s * sin(angle));
    }
    glEnd();
    
    // Swap buffers (double buffering)
    glutSwapBuffers();
}

void screenToGL(int x, int y, float &glX, float &glY) {
    // Convert screen coordinates to OpenGL coordinates
    glX = (2.0f * x / glutGet(GLUT_WINDOW_WIDTH)) - 1.0f;
    glY = 1.0f - (2.0f * y / glutGet(GLUT_WINDOW_HEIGHT));
}

void update(int value) {
    // Animation logic
    if (animationEnabled) {
        cx += dx;
        cy += dy;
        
        // Bounce off walls
        if (cx + s > 1.0f || cx - s < -1.0f) dx = -dx;
        if (cy + s > 1.0f || cy - s < -1.0f) dy = -dy;
    }
    
    // Request redisplay
    glutPostRedisplay();
    
    // Set up next timer
    glutTimerFunc(16, update, 0);  // ~60fps
}


void mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        float glX, glY;
        screenToGL(x, y, glX, glY);
        
        if (button == GLUT_LEFT_BUTTON) {
            // Set position
            cx = glX;
            cy = glY;
        }
        else if (button == GLUT_RIGHT_BUTTON) {
            // Set velocity
            dx = (cx - glX) * 0.02f;
            dy = (cy - glY) * 0.02f;
        }
        
        glutPostRedisplay();
    }
}

void mouseMotion(int x, int y) {
    float glX, glY;
    screenToGL(x, y, glX, glY);
    
    // Update position
    cx = glX;
    cy = glY;
    
    glutPostRedisplay();
}

void init() {
    // Set background color (white)
    glClearColor(1.0, 1.0, 1.0, 1.0);
    
    // Set up projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);
}

int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutCreateWindow("Circle with Mouse Control");
    
    // Register callbacks
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMotion);
    glutTimerFunc(16, update, 0);
    
    // Initialize OpenGL settings
    init();
    
    // Enter the main loop
    glutMainLoop();
    return 0;
}
```

### Mouse Control Features
- **Left Click**: Set circle position directly to the clicked location
- **Right Click**: Set velocity direction based on the clicked position relative to circle
- **Click and Drag**: Move the circle by dragging with the mouse
- **Automatic Animation**: Can be enabled to make the circle bounce off screen edges

## Animation Example

```cpp
#include <iostream>
#include <GL/glut.h>
#include <cmath>

// Circle properties
float cx = 0.0f, cy = 0.0f;    // Center position
float s = 0.2f;                // Size/scale

// Animation properties
float dx = 0.01f, dy = 0.01f;  // Velocity
bool animationEnabled = false; // Animation starts paused

void display() {
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Draw red circle
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POLYGON);
    for (int i = 0; i < 360; i++) {
        float angle = i * M_PI / 180.0f;
        glVertex2f(cx + s * cos(angle), cy + s * sin(angle));
    }
    glEnd();
    
    // Swap buffers (double buffering)
    glutSwapBuffers();
}

void update(int value) {
    // Animation logic
    if (animationEnabled) {
        cx += dx;
        cy += dy;
        
        // Bounce off walls
        if (cx + s > 1.0f || cx - s < -1.0f) dx = -dx;
        if (cy + s > 1.0f || cy - s < -1.0f) dy = -dy;
    }
    
    // Request redisplay
    glutPostRedisplay();
    
    // Set up next timer
    glutTimerFunc(16, update, 0);  // ~60fps
}

void keyboard(unsigned char key, int x, int y) {
    // Process keyboard input
    switch(key) {
        case 27:  // ESC key
            exit(0);
            break;
        case ' ':  // Space key
            animationEnabled = !animationEnabled;
            break;
        case 'r':  // Reset position
            cx = 0.0f;
            cy = 0.0f;
            break;
    }
    
    glutPostRedisplay();
}

void init() {
    // Set background color (white)
    glClearColor(1.0, 1.0, 1.0, 1.0);
    
    // Set up projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);
}

int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutCreateWindow("Circle Animation");
    
    // Register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(16, update, 0);
    
    // Initialize OpenGL settings
    init();
    
    // Enter the main loop
    glutMainLoop();
    return 0;
}
```

### Animation Features
- **Space**: Toggle animation on/off
- **R**: Reset circle position to center
- **ESC**: Exit the program
- **Physics**: Circle bounces off screen edges when animation is enabled
- **Timer**: Updates at approximately 60fps using GLUT's timer function

## Keyboard Control Example

```cpp
#include <iostream>
#include <GL/glut.h>
#include <cmath>

// Circle properties
float cx = 0.0f, cy = 0.0f;    // Center position
float s = 0.2f;                // Size/scale

// Animation properties
float dx = 0.01f, dy = 0.01f;  // Velocity
bool animationEnabled = false; // Animation starts paused

// Keyboard state tracking
bool keys[256] = {false};
bool specialKeys[256] = {false};

void display() {
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Draw red circle
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POLYGON);
    for (int i = 0; i < 360; i++) {
        float angle = i * M_PI / 180.0f;
        glVertex2f(cx + s * cos(angle), cy + s * sin(angle));
    }
    glEnd();
    
    // Swap buffers (double buffering)
    glutSwapBuffers();
}

void processMovement() {
    float speed = 0.02f;
    float newX = cx, newY = cy;
    
    // Process WASD & arrow keys
    if ((keys['w'] || keys['W'] || specialKeys[GLUT_KEY_UP])) newY += speed;
    if ((keys['s'] || keys['S'] || specialKeys[GLUT_KEY_DOWN])) newY -= speed;
    if ((keys['a'] || keys['A'] || specialKeys[GLUT_KEY_LEFT])) newX -= speed;
    if ((keys['d'] || keys['D'] || specialKeys[GLUT_KEY_RIGHT])) newX += speed;
    
    // Apply movement if within bounds
    if (newX + s <= 1.0f && newX - s >= -1.0f) cx = newX;
    if (newY + s <= 1.0f && newY - s >= -1.0f) cy = newY;
}

void update(int value) {
    // Process keyboard input
    processMovement();
    
    // Animation logic
    if (animationEnabled) {
        cx += dx;
        cy += dy;
        
        // Bounce off walls
        if (cx + s > 1.0f || cx - s < -1.0f) dx = -dx;
        if (cy + s > 1.0f || cy - s < -1.0f) dy = -dy;
    }
    
    // Request redisplay
    glutPostRedisplay();
    
    // Set up next timer
    glutTimerFunc(16, update, 0);  // ~60fps
}

void keyboard(unsigned char key, int x, int y) {
    keys[key] = true;
    
    // Handle immediate key actions
    switch(key) {
        case 27:  exit(0); break;               // ESC to exit
        case 'r': cx = cy = 0.0f; break;        // Reset position
        case ' ': animationEnabled = !animationEnabled; break;  // Toggle animation
        case '+': s += 0.05f; break;            // Increase size
        case '-': s = (s > 0.05f) ? s - 0.05f : s; break;  // Decrease size
    }
    
    glutPostRedisplay();
}

void keyboardUp(unsigned char key, int x, int y) {
    keys[key] = false;
}

void specialKeyboard(int key, int x, int y) {
    specialKeys[key] = true;
}

void specialKeyboardUp(int key, int x, int y) {
    specialKeys[key] = false;
}

void init() {
    // Set background color (white)
    glClearColor(1.0, 1.0, 1.0, 1.0);
    
    // Set up projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);
}

int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutCreateWindow("Circle with Keyboard Control");
    
    // Register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboardUp);
    glutSpecialFunc(specialKeyboard);
    glutSpecialUpFunc(specialKeyboardUp);
    glutTimerFunc(16, update, 0);
    
    // Initialize OpenGL settings
    init();
    
    // Enter the main loop
    glutMainLoop();
    return 0;
}
```

### Keyboard Control Features
- **WASD/Arrow Keys**: Move the circle in four directions
- **Space**: Toggle animation on/off
- **R**: Reset circle position to center
- **+/-**: Increase/decrease circle size
- **ESC**: Exit the program
- **Key State Tracking**: Maintains state of pressed keys for smooth movement
- **Boundary Checking**: Prevents circle from moving outside visible area
