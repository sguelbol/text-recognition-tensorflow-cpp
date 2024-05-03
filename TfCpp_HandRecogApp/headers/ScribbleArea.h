#ifndef SCRIBBLEAREA_H
#define SCRIBBLEAREA_H

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
#include <QColor>
#include <QImage>
#include <QPoint>
#include <QWidget>
#include <opencv2/opencv.hpp>

class ScribbleArea : public QWidget
{
    Q_OBJECT
public:
    std::vector<std::vector<int>> inputs;
    int x_min, x_max, y_min, y_max;
    ScribbleArea(QWidget *parent = 0);
    void setModel(const std::shared_ptr<Model> model);
    bool openImage(const QString &fileName);
    bool saveImage(const QString &fileName, const char *fileFormat);
    void setPenColor(const QColor &newColor);
    void setPenWidth(int newWidth);
    void boundingBox();

    bool isModified() const {return modified;}
    QColor penColor() const {return myPenColor;}
    int penWidth() const {return myPenWidth;}



    public slots:
        void clearImage();
    void print();
    void transform(QImage *image);


protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    std::shared_ptr<Model> model;
    void drawLineTo(const QPoint &endPoint);
    void resizeImage(QImage *image, const QSize &newSize);
    bool modified;
    bool scribbling;
    QColor myPenColor;
    int myPenWidth;
    QImage image;
    QPoint lastPoint;
    void transformTo28(QImage *letter);
    cv::Mat QImageToCvMat(const QImage &image);

    QImage cvMatToQImage(const cv::Mat &inMat);
};

#endif //SCRIBBLEAREA_H
