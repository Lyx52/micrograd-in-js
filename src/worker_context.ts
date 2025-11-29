import * as Comlink from 'comlink';
import type {RemoteObject} from "comlink";
import type {NetworkLayer} from "./render/NetworkLayer.ts";
import {LinearModule} from "./nn/module/linear_module.ts";
import type {NewTensor} from "./nn/tensor_new.ts";
export interface INetworkWorker {
    module?: LinearModule;
    layers: NetworkLayer[];
    loss: number[];
    xs: NewTensor[];
    ys: NewTensor[];
    epochs: number;
    learningRate: number;
    seed: number;
    lossEveryN: number;
    batchSize: number;
    training: boolean;
    createModule(layers: NetworkLayer[], seed: number);
    startTraining(epochs: number, learningRate: number, lossEveryN: number, x: number[][], y: number[][], batchSize: number);
    startTrainingMinst(epochs: number, learningRate: number, lossEveryN: number, batchSize: number);
    testModule(xs: number[]): number[];
    cancelTraining();
    resetModule();
    publishLoss();
    trainModule();
}

export interface IWorker {
    comlink: RemoteObject<INetworkWorker>,
    worker: Worker,
}

export const createWorker = (): IWorker => {
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    const comlink = Comlink.wrap<INetworkWorker>(worker) as RemoteObject<INetworkWorker>;

    return {
        comlink,
        worker,
    } as IWorker;
};