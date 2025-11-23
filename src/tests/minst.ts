import {readIdx} from "../idx.ts";
import {trainImages, trainLabels} from "../../data/data.ts";
import {Tensor} from "../nn/tensor.ts";
import {Value} from "../nn/value.ts";
import {LinearModule} from "../nn/linear_module.ts";
import {Flatten} from "../nn/flatten.ts";
import {Linear} from "../nn/linear.ts";
import {ReLu} from "../nn/relu.ts";
import {Softmax} from "../nn/softmax.ts";
import {crossEntropyLoss} from "../nn/utils.ts";


export const runMinst = async (trainCount = 5, epochs = 500, learningRate = 0.25) => {
    const [_, labelsData] = await readIdx(trainLabels, [trainCount]);
    const [imagesDims, imagesData] = await readIdx(trainImages, [trainCount, -1, -1]);

    const ys = labelsData
        .map(v => new Array(10).fill(0).map((_, i) => v === (i + 1) ? 1 : 0))
        .map(v => Tensor.fromValues(...v).setDimensions([10]));

    const trainImageData = imagesData.map(v => new Value(v))
    let imgSize = imagesDims[1] * imagesDims[2];
    const xs = [];
    for (let i = 0; i < imagesDims[0]; i++) {
        xs.push(Tensor.from(trainImageData.slice(i * imgSize, i * imgSize + imgSize)).setDimensions([imagesDims[1], imagesDims[2]]));
    }
    console.log(`Dataset loaded ${trainCount} labels/images`)

    const module = new LinearModule(
        new Flatten(),
        new Linear(28 * 28, 32),
        new ReLu(),
        new Linear(32, 16),
        new Linear(16, 10),
        new Softmax(),
    );

    for (let i = 0; i <= epochs; i++) {
        console.log(`[Step ${i}/${epochs}] starting...`)
        for (const parameter of  module.parameters()) {
            parameter.Data += -learningRate * parameter.Grad;
        }

        const loss = crossEntropyLoss(module, ys, xs).scalar();
        module.zerograd();
        loss.backward();

        console.log(`[Step ${i}/${epochs}] loss: ${loss.Data}`)
    }
}

