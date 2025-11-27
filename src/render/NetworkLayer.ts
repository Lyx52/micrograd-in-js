import {LayerType, toLayerName} from "./LayerTypes.ts";
import type {ICallable} from "../nn/interfaces/ICallable.ts";
import {Flatten} from "../nn/layers/flatten.ts";
import {ReLu} from "../nn/layers/relu.ts";
import {Linear} from "../nn/layers/linear.ts";
import {Softmax} from "../nn/mlp/softmax.ts";

export class NetworkLayer {
    constructor(public id: number, public type: LayerType, public inputs: number, public outputs: number, public useBias: boolean = false) {

    }
    public typeName() {
        return toLayerName(this.type);
    }

    public toLayer(): ICallable {
        switch (this.type) {
            case LayerType.Flatten: return new Flatten();
            case LayerType.ReLu: return new ReLu();
            case LayerType.Linear: return new Linear(this.inputs, this.outputs, this.useBias);
            case LayerType.Softmax: throw new Softmax();
        }
    }
}