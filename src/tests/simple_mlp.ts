import {MLP} from "../nn/mlp.ts";
import {forward, renderGraph} from "../nn/utils.ts";
import {Tensor} from "../nn/tensor.ts";

export const runSimpleMLP = () => {
    const mlp = new MLP(3, [4, 1]);
    const xs = [
        Tensor.fromValues(2.0, 3.0, -1.0),
        Tensor.fromValues(3.0, -1.0, 0.5),
        Tensor.fromValues(0.5, 1.0, 1.0),
        Tensor.fromValues(1.0, 1.0, -1.0)
    ];
    const ys = [
        Tensor.fromValues(1.0),
        Tensor.fromValues(-1.0),
        Tensor.fromValues(-1.0),
        Tensor.fromValues(1.0),
    ];

    // Train
    const learningRate = 0.15;
    const steps = 20;
    let lossx = null;
    for (let i = 0; i <= steps; i++) {
        for (const parameter of  mlp.parameters()) {
            parameter.Data += -learningRate * parameter.Grad;
        }

        const loss = forward(mlp, ys, xs);
        const lossItem = loss.item().shift();
        mlp.zerograd();
        lossItem.backward();

        console.log(`[Step ${i}/${steps}] loss: ${lossItem.Data}`)
        lossx = lossItem;
    }

    renderGraph(lossx);
}