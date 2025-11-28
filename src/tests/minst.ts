import {readIdx} from "../idx.ts";
import {trainImages, trainLabels} from "../../data/data.ts";
import {Tensor} from "../nn/tensor.ts";
import {Value} from "../nn/value.ts";
import {LinearModule} from "../nn/module/linear_module.ts";
import {Flatten} from "../nn/layers/flatten.ts";
import {Linear} from "../nn/layers/linear.ts";
import {ReLu} from "../nn/layers/relu.ts";
import {Softmax} from "../nn/layers/softmax.ts";
import {crossEntropyLoss, maxMarginLoss} from "../nn/utils.ts";
import {NewTensor} from "../nn/tensor_new.ts";


export const runMinst = async (trainCount = 3, epochs = 1, learningRate = 0.01) => {
    const [_, labelsData] = await readIdx(trainLabels, [trainCount]);
    const [imagesDims, imagesData] = await readIdx(trainImages, [trainCount, -1, -1]);

    const ys = labelsData
        .map(v => new Array(10).fill(0).map((_, i) => v === i ? 1 : 0))
        .map(v => NewTensor.from(v));

    let imgSize = imagesDims[1] * imagesDims[2];
    const xs: NewTensor[] = [];
    for (let i = 0; i < imagesDims[0]; i++) {
        xs.push(NewTensor.from(imagesData.slice(i * imgSize, i * imgSize + imgSize)).setDimension([imagesDims[1], imagesDims[2]]));
    }
    console.log(`Dataset loaded ${trainCount} labels/images`)

    const sampleRandom = () => {
        const randomIndexes = new Array(10).fill(0).map((_, i) => Math.floor(Math.random() * ys.length));
        return [
            randomIndexes.map(i => ys[i]),
            randomIndexes.map(i => xs[i]),
        ]
    }

    const labelToNumber = (values: number[]) => {
        let maxIndex = 0;
        for (let i = 0; i < values.length; i++) {
            if (values[maxIndex] < values[i]) {
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    const module = new LinearModule(
        new Flatten(),
        new Linear(28 * 28, 64),
        new ReLu(),
        new Linear(64, 32),
        new Linear(32, 10),
    );

    for (let i = 0; i <= epochs; i++) {
        console.time('TrainingStep')
        module.updateParameters((tensor: NewTensor) => {
            for (let j = 0; j < tensor.backing.length; j++) {
                tensor.backing[j] += -learningRate * tensor.gradients[j];
            }
        });

        const loss = crossEntropyLoss(module, ys, xs);
        module.zerograd();
        loss.backward();
        console.timeEnd('TrainingStep')

        if ((i % 10) === 0) {
            const [labels, images] = sampleRandom();
            for (let i = 0; i < labels.length; i++) {
                const guess = module.execute(images[i]);
                console.log(`[Guess ${i}] GUESS: ${labelToNumber(guess.backing)}, EXPECTED: ${labelToNumber(labels[i].backing)}`);
            }
        }
        console.log(`[Step ${i}/${epochs}] loss: ${loss.scalar()}`)
    }
}

