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
