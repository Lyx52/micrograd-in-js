import * as Comlink from 'comlink';
import type {RemoteObject} from "comlink";
import type {NetworkLayer} from "./render/NetworkLayer.ts";
import {LinearModule} from "./nn/module/linear_module.ts";
export interface INetworkWorker {
    module?: LinearModule;
    layers: NetworkLayer[];
    loss: number[];
    xs: number[][];
    ys: number[][];
    epochs: number;
    learningRate: number;
    seed: number;
    createModule(layers: NetworkLayer[], seed: number);
    startTraining(epochs: number, learningRate: number, lossEveryN: number, x: number[][], y: number[][]);
    startTrainingMinst(epochs: number, learningRate: number, lossEveryN: number);
    testModule(xs: number[]): number[];
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