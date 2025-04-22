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
