import {LayerType, toLayerName} from "./LayerTypes.ts";

export class NetworkLayer {
    constructor(public id: number, public type: LayerType, public inputs: number, public outputs: number, public useBias: boolean = false) {

    }
    public typeName() {
        return toLayerName(this.type);
    }
}