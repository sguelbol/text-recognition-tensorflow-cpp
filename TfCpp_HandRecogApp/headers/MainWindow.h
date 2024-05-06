#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#undef signals
#define singals tf_signals
#define quint8 tf_quint8
#define qint8 tf_qint8
#define quint16 tf_quint16
#define qint16 tf_qint16
#define qint32 tf_qint32
#include <Model.h>
#undef quint8
#undef qint8
#undef quint16
#undef qint16
#undef qint32
#undef signals


#include <QMainWindow>
#include <QPushButton>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QList>
#include "ScribbleArea.h"

using namespace std;

class ScribbleArea;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    void addModel(shared_ptr<Model> model);

private:
    void createActions();
    void createMenus();
    void createTrainingField(QVBoxLayout &mainLayout);
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

    QPushButton *button;
    QLineEdit *textField;

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void open();
    void save();
    void penColor();
    void penWidth();
    void about();
    void retrain();

};
#endif //MAINWINDOW_H
