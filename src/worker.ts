import * as Comlink from 'comlink';
import type {INetworkWorker} from "./worker_context.ts";
import type {NetworkLayer} from "./render/NetworkLayer.ts";
import {LinearModule} from "./nn/module/linear_module.ts";
import type {ICallable} from "./nn/interfaces/ICallable.ts";
import {LayerType} from "./render/LayerTypes.ts";
import {Flatten} from "./nn/layers/flatten.ts";
import {ReLu} from "./nn/layers/relu.ts";
import {Linear} from "./nn/layers/linear.ts";
import {Softmax} from "./nn/layers/softmax.ts";
import {NewTensor} from "./nn/tensor_new.ts";
import {crossEntropyLoss} from "./nn/utils.ts";
import {RandomGenerator} from "./nn/random.ts";
import {readIdx} from "./idx.ts";
import {trainImages, trainLabels} from "../data/data.ts";

const toLayer = (layer: NetworkLayer): ICallable => {
    switch (layer.type) {
        case LayerType.Flatten: return new Flatten();
        case LayerType.ReLu: return new ReLu();
        case LayerType.Linear: return new Linear(layer.inputs, layer.outputs);
        case LayerType.Softmax: return new Softmax();
    }
}
const layersEqual = (first: NetworkLayer, second: NetworkLayer) => {
    for (const [property, value] of Object.entries(first)) {
        if (second[property] !== value) {
            return false;
        }
    }

    return true;
}

const allLayersEqual = (first: NetworkLayer[], second: NetworkLayer[]) => {
    if (first.length !== second.length) {
        return false;
    }

    for (let i = 0; i < first.length; i++) {
        if (!layersEqual(first[i], second[i])) {
            return false;
        }
    }

    return true;
}

const worker: INetworkWorker = {
    module: undefined,
    layers: [],
    loss: [],
    xs: [],
    ys: [],
    epochs: 100,
    learningRate: 0.01,
    seed: 0,
    lossEveryN: 3,
    createModule(layers: NetworkLayer[], seed = 0) {
        this.seed = seed;
        this.loss = [];
        this.layers = layers;
        this.resetModule();
    },
    resetModule() {
        RandomGenerator.Seed(this.seed);
        this.module = new LinearModule(...this.layers.map(v => toLayer(v)));
    },
    publishLoss() {
        postMessage({
            type: 'LOSS',
            data: this.loss
        })
    },
    async startTrainingMinst(epochs: number, learningRate: number, lossEveryN: number) {
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.loss = [];
        this.lossEveryN = lossEveryN;

        const [_, labelsData] = await readIdx(trainLabels, [100]);
        const [imagesDims, imagesData] = await readIdx(trainImages, [100, -1, -1]);

        this.ys = labelsData
            .map(v => new Array(10).fill(0).map((_, i) => v === i ? 1 : 0))
            .map(v => NewTensor.from(v));

        let imgSize = imagesDims[1] * imagesDims[2];
        this.xs = [];
        for (let i = 0; i < imagesDims[0]; i++) {
            this.xs.push(NewTensor.from(imagesData.slice(i * imgSize, i * imgSize + imgSize)).setDimension([imagesDims[1], imagesDims[2]]));
        }

        this.trainModule();
    },
    startTraining(epochs: number, learningRate: number, lossEveryN: number, x: number[][], y: number[][]) {
        this.epochs = epochs;
        this.learningRate = learningRate;

        if (x.length !== y.length) {
            return;
        }

        if (!this.module) {
            return;
        }

        this.loss = [];
        this.lossEveryN = lossEveryN;
        this.xs = x.map(v => NewTensor.from(v));
        this.ys = y.map(v => NewTensor.from(v));

        this.trainModule();
    },
    testModule(xs: number[]): number[] {
        return this.module?.execute(NewTensor.from(xs))?.backing ?? [];
    },
    trainModule() {
        if (!this.module) {
            return;
        }

        for (let i = 0; i <= this.epochs; i++) {
            this.module.updateParameters((tensor: NewTensor) => {
                for (let j = 0; j < tensor.backing.length; j++) {
                    tensor.backing[j] += -this.learningRate * tensor.gradients[j];
                }
            });

            const loss = crossEntropyLoss(this.module, this.ys, this.xs);
            this.module.zerograd();
            loss.backward();

            if ((i % this.lossEveryN) === 0) {
                this.loss.push(loss.scalar());
                this.publishLoss();
            }
        }
    }
};

Comlink.expose(worker);