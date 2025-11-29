import {LayerType, toLayerName} from "./LayerTypes.ts";
import type {ActivationType} from "./ActivationTypes.ts";

export class NetworkLayer {
    constructor(public id: number, public type: LayerType, public activation: ActivationType, public inputs: number, public outputs: number, public useBias: boolean = false) {

    }
    public typeName() {
        return toLayerName(this.type);
    }
}