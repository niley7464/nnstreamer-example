/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file main.js
 * @date 30 April 2024
 * @brief Image classification Offloading example
 * @author Yelin Jeong <yelini.jeong@samsung.com>
 */

import {
  getNetworkType,
  getIpAddress,
  GetMaxIdx,
  GetImgPath,
  loadLabelInfo,
  startHybridService,
  startMessagePort,
  gRemoteServices,
} from "./utils.js";

const serviceName = "mobilenet_v1_1.0_224_quant";
let fHandle = null;
let tensorsData = null;
let tensorsInfo = null;

function disposeData() {
  if (fHandle != null) {
    fHandle.close();
  }

  if (tensorsData != null) {
    tensorsData.dispose();
  }

  if (tensorsInfo != null) {
    tensorsInfo.dispose();
  }
}

let localHandle;
let offloadingHandle;

function createPipelineDescription(isLocal, filter) {
  return (
    "appsrc caps=image/jpeg name=srcx_" +
    (isLocal ? "local" : "offloading") +
    " ! jpegdec ! " +
    "videoconvert ! video/x-raw,format=RGB,framerate=0/1,width=224,height=224 ! tensor_converter ! " +
    "other/tensors,num_tensors=1,format=static,dimensions=(string)3:224:224:1,types=uint8,framerate=0/1  ! " +
    filter +
    " ! " +
    "other/tensors,num_tensors=1,format=static,dimensions=(string)1001:1,types=uint8,framerate=0/1 ! " +
    "tensor_sink name=sinkx_" +
    (isLocal ? "local" : "offloading")
  );
}

/**
 * Callback function for pipeline sink listener
 */
function sinkListenerCallback(sinkName, data) {
  const endTime = performance.now();

  const tensorsRetData = data.getTensorRawData(0);
  const maxIdx = GetMaxIdx(tensorsRetData.data);

  let type;
  if (sinkName.endsWith("local")) {
    type = "local";
  } else {
    type = "offloading";
  }

  const label = document.getElementById("label_" + type);
  label.innerText = labels[maxIdx];

  const time = document.getElementById("time_" + type);
  time.innerText = type + " : " + (endTime - startTime).toFixed(3) + " ms";
}

/**
 * Run a pipeline that uses Tizen device's resources
 */
function runLocal() {
  const modelPath = "wgt-package/res/mobilenet_v1_1.0_224_quant.tflite";
  const URI_PREFIX = "file://";
  const absModelPath = tizen.filesystem
    .toURI(modelPath)
    .substr(URI_PREFIX.length);
  const filter =
    "tensor_filter framework=tensorflow-lite model=" + absModelPath;

  const pipelineDescription = createPipelineDescription(true, filter);

  localHandle = tizen.ml.pipeline.createPipeline(pipelineDescription);
  localHandle.start();
  localHandle.registerSinkListener("sinkx_local", sinkListenerCallback);
}

/**
 * Run a pipeline that uses other device's resources
 */
function runOffloading() {
  const remoteService = gRemoteServices.get(serviceName);
  if (
    !gRemoteServices.has(serviceName) ||
    !Object.prototype.hasOwnProperty.call(remoteService, "ip") ||
    !Object.prototype.hasOwnProperty.call(remoteService, "port")
  ) {
    console.log("No remote service available");
    return;
  }

  const filter =
    "tensor_query_client host=" +
    ip +
    " port=" +
    remoteService["port"] +
    " dest-host=" +
    remoteService["ip"] +
    " dest-port=" +
    remoteService["port"] +
    " timeout=1000";

  const pipelineDescription = createPipelineDescription(false, filter);

  offloadingHandle = tizen.ml.pipeline.createPipeline(pipelineDescription);
  offloadingHandle.start();
  offloadingHandle.registerSinkListener(
    "sinkx_offloading",
    sinkListenerCallback,
  );
}

let startTime;

/**
 * Run a pipeline that uses other device's resources
 */
function inference(isLocal) {
  if (!isLocal && !gRemoteServices.has(serviceName)) {
    console.log("Offloading service is disappeared");
    return;
  }

  const img_path = GetImgPath();
  let img = new Image();
  img.src = img_path;

  img.onload = function () {
    disposeData();
    fHandle = tizen.filesystem.openFile("wgt-package" + img_path, "r");
    const imgUInt8Array = fHandle.readData();

    tensorsInfo = new tizen.ml.TensorsInfo();
    tensorsInfo.addTensorInfo("tensor", "UINT8", [imgUInt8Array.length]);
    tensorsData = tensorsInfo.getTensorsData();
    tensorsData.setTensorRawData(0, imgUInt8Array);

    startTime = performance.now();

    if (isLocal) {
      localHandle.getSource("srcx_local").inputData(tensorsData);
    } else {
      offloadingHandle.getSource("srcx_offloading").inputData(tensorsData);
    }

    const canvasName = "canvas_" + (isLocal ? "local" : "offloading");
    const canvas = document.getElementById(canvasName);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
  };
}

let ip;
let labels;

window.onload = async function () {
  const networkType = await getNetworkType();
  ip = await getIpAddress(networkType);
  labels = loadLabelInfo();
  startMessagePort();
  startHybridService();
  document.getElementById("start_local").addEventListener("click", function () {
    runLocal();
  });

  document
    .getElementById("start_offloading")
    .addEventListener("click", function () {
      runOffloading();
    });

  document.getElementById("local").addEventListener("click", function () {
    inference(true);
  });

  document.getElementById("offloading").addEventListener("click", function () {
    inference(false);
  });

  /* add eventListener for tizenhwkey */
  document.addEventListener("tizenhwkey", function (e) {
    if (e.keyName === "back") {
      try {
        console.log("Pipeline is disposed!!");
        localHandle.stop();
        localHandle.dispose();

        offloadingHandle.stop();
        offloadingHandle.dispose();

        disposeData();

        tizen.application.getCurrentApplication().exit();
      } catch (ignore) {
        console.log("error " + ignore);
      }
    }
  });
};
