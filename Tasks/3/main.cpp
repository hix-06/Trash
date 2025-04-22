#include <GL/glut.h>  // Include the GLUT library
#include <cmath>      // Include for mathematical functions

// Global variables (if needed)
// None required for this example

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
