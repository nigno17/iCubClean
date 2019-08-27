/*
 * Copyright: (C) 2012-2015 POETICON++, European Commission FP7 project ICT-288382
 * Copyright: (C) 2014 VisLab, Institute for Systems and Robotics,
 *                Instituto Superior Técnico, Universidade de Lisboa, Lisbon, Portugal
 * Author: Afonso Gonçalves <agoncalves@isr.ist.utl.pt>
 * CopyPolicy: Released under the terms of the GNU GPL v2.0
 *
 */

#include "perspectiveChanger.h"

using namespace std;
using namespace yarp::os;
using namespace yarp::sig;
using namespace iCub::ctrl;
using namespace yarp::math;

BufferedPort<Bottle> port;


double PerspectiveChangerModule::getPeriod() {

    return 1;
}

bool   PerspectiveChangerModule::updateModule() {

    return true;
}

bool   PerspectiveChangerModule::configure(yarp::os::ResourceFinder &rf) {

    /* module name */
    moduleName = rf.check("name", Value("perspectiveChanger"),
                          "Module name (string)").asString();

    setName(moduleName.c_str());

    /* port names */
    perspectiveChangerPortName  = "/";
    perspectiveChangerPortName += getName( rf.check("perspectiveChangerPort",
                                     Value("/img:io"),
                                     "Changer port(string)")
                                     .asString() );

    perspectiveChangerTransformationPortName  = "/";
    perspectiveChangerTransformationPortName += getName( rf.check("perspectiveChangerTransformationPort",
                                     Value("/matrix:o"),
                                     "Changer port(string)")
                                     .asString() );

    /* open ports */
    if (!perspectiveChangerPort.open(
            perspectiveChangerPortName.c_str()))
    {
        cout << getName() << ": unable to open port"
        << perspectiveChangerPortName << endl;
        return false;
    }

    if (!perspectiveChangerTransformationPort.open(
            perspectiveChangerTransformationPortName.c_str()))
    {
        cout << getName() << ": unable to open port"
        << perspectiveChangerTransformationPortName << endl;
        return false;
    }

    /* Rate thread period */
    threadPeriod = rf.check("threadPeriod", Value(0.033),
        "Control rate thread period key value(double) in seconds ").asDouble();

    /* Table offset */
    tableOffset = rf.check("table", Value(-0.1),
        "Table height relative to the robot reference frame (double) in meters ").asDouble();

    /* Pixel density */
    pixelDensity = rf.check("ppm", Value(240),
        "Pixels per Meter, number of pixels that represent a meter in the horizontal plane of the table").asDouble();

    /* Create the control rate thread */
    ctrlThread = new CtrlThread(&perspectiveChangerPort,
                                &perspectiveChangerTransformationPort,
                                tableOffset,
                                pixelDensity,
                                threadPeriod);

    /* Starts the thread */
    if (!ctrlThread->start()) {
        delete ctrlThread;
        return false;
    }

    return true;
}

bool   PerspectiveChangerModule::interruptModule() {

    cout << "Interrupting your module, for port cleanup" << endl;

    perspectiveChangerPort.interrupt();
    perspectiveChangerTransformationPort.interrupt();

    return true;
}

bool   PerspectiveChangerModule::close() {

    /* optional, close port explicitly */
    cout << "Calling close function\n";

    ctrlThread->stop();
    delete ctrlThread;
    perspectiveChangerPort.close();
    perspectiveChangerTransformationPort.close();

    return true;
}

CtrlThread::CtrlThread(BufferedPort<ImageOf<PixelBgr> > *modulePort,
                       BufferedPort<Bottle> *moduleTransformationPort,
                       const double table,
                       const double pixelD,
                       const double period)
                       :RateThread(int(period*1000.0)) {

    perspectiveChangerPort = modulePort;
    perspectiveChangerTransformationPort = moduleTransformationPort;
    tableOffset = table;
    pixelDensity = pixelD;
}

bool CtrlThread::threadInit() {

    cout << endl << "thread initialization" << endl;

    Property optGaze("(device gazecontrollerclient)");
    optGaze.put("remote","/iKinGazeCtrl");
    optGaze.put("local", "/gaze_client");
    port.open("/invLambda:o");

    if (!clientGazeCtrl.open(optGaze)) {
        return false;
    }

    if (clientGazeCtrl.isValid()) {
        clientGazeCtrl.view(igaze);
    }

    igaze->storeContext(&startupContextId);

    return true;
}

void CtrlThread::threadRelease() {

    cout << endl << "thread release" << endl;

    igaze->stopControl();
    igaze->restoreContext(startupContextId);
    clientGazeCtrl.close();
}

void CtrlThread::run() {

    ImageOf<PixelBgr> *yarpImage = perspectiveChangerPort->read();

    if (yarpImage!=NULL) {
        Vector  plane(4);
        Vector  cartCoord0(3);
        Vector  imgCoord0(2);
        Vector  pixel(2);
        std::vector<Vector> cartCoord(4, cartCoord0);
        std::vector<Vector> imgCoord(4, imgCoord0);
        cv::Point2f inputQuad[4];
        cv::Point2f outputQuad[4];
        cv::Mat lambda;//(3, 3, CV_32FC1);
        cv::Mat invLambda;
        Bottle invLambdaBottle;
        ImageOf<PixelBgr> tempImg;

        cv::Mat imgMat( (IplImage*)yarpImage->getIplImage() );
        cv::Mat imgMat2;

        Vector x;
        Vector px;
        px.resize(2);
        x.resize(3);
        int camsel = 0; // 0 left, 1 right


        double depth = 1.0;
        double width = depth * (2.0 / 3.0);
        //x1
        x(0) = -depth; x(1) = -width; x(2) = tableOffset;
        igaze->get2DPixel(camsel, x, px);
        inputQuad[0] = cv::Point2f(px(0),px(1));
        cout << "0x" << px.toString().c_str();
        //x2
        x(0) = -depth; x(1) = width; x(2) = tableOffset;
        igaze->get2DPixel(camsel, x, px);
        inputQuad[1] = cv::Point2f(px(0),px(1));
        cout << "1x" << px.toString().c_str();
        //x3
        x(0) = 0; x(1) = width; x(2) = tableOffset;
        igaze->get2DPixel(camsel, x, px);
        inputQuad[2] = cv::Point2f(px(0),px(1));
        cout << "2x" << px.toString().c_str();
        //x4
        x(0) = 0; x(1) = -width; x(2) = tableOffset;
        igaze->get2DPixel(camsel, x, px);
        inputQuad[3] = cv::Point2f(px(0),px(1));
        cout << "3x" << px.toString().c_str();
        
        outputQuad[0] = cv::Point2f(             0,             0 );
        outputQuad[1] = cv::Point2f( imgMat.cols-1,             0 );
        outputQuad[2] = cv::Point2f( imgMat.cols-1, imgMat.rows-1 );
        outputQuad[3] = cv::Point2f(             0, imgMat.rows-1 );
        
        lambda = cv::getPerspectiveTransform( inputQuad, outputQuad );
        warpPerspective( imgMat, imgMat2, lambda, imgMat.size() );


        tempImg.setExternal( imgMat2.data, imgMat2.size[1], imgMat2.size[0] );
        perspectiveChangerPort->prepare() = tempImg;
        perspectiveChangerPort->write();

        invLambda = lambda.inv();
        //cout << "++++++++" << invLambda << endl;
        Bottle& output = port.prepare();
        output.clear();
        for(int row = 0; row < invLambda.rows; row++) {
            for(int col = 0; col < invLambda.cols; col++) {
                invLambdaBottle.addDouble(invLambda.at<double>(row, col));
                output.addDouble(invLambda.at<double>(row, col));
                cout << "++++++++" << invLambda.at<double>(row, col) << endl;
            }
        }
        port.write();

        perspectiveChangerTransformationPort->prepare() = invLambdaBottle;
        perspectiveChangerTransformationPort->write();
    }
}
