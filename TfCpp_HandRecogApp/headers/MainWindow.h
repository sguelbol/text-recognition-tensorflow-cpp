#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QList>
#include "ScribbleArea.h"

class ScribbleArea;


class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);


private:
    void createActions();
    void createMenus();
    bool maybeSave();
    bool saveFile(const QByteArray &fileFormat);
    ScribbleArea *scribbleArea;

    //Menu widgets
    QMenu *saveAsMenu;
    QMenu *fileMenu;
    QMenu *optionMenu;
    QMenu *helpMenu;

    //all the actions that can occur
    QAction *openAct;

    //actions tied to specific file format
    QList<QAction *> saveAsActs;
    QAction *exitAct;
    QAction *penColorAct;
    QAction *penWidthAct;
    QAction *printAct;
    QAction *clearScreenAct;
    QAction *aboutAct;
    QAction *aboutQtAct;



protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void open();
    void save();
    void penColor();
    void penWidth();
    void about();

};
#endif //MAINWINDOW_H
