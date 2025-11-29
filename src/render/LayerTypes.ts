import type {IOption} from "../nn/utils.ts";

export enum LayerType {
    Linear = 'linear',
    ReLu = 'relu',
    Flatten = 'flatten',
    Softmax = 'softmax',
}

export function toLayerName(type: LayerType) {
    switch (type) {
        case LayerType.Linear:
            return 'Linear';
        case LayerType.ReLu:
            return 'ReLu';
        case LayerType.Flatten:
            return 'Flatten';
        case LayerType.Softmax:
            return 'Softmax';
        default:
            return '';
    }
}

export const getLayerOptions = (): IOption<LayerType>[] => {
    const options = [LayerType.Linear, LayerType.ReLu, LayerType.Flatten, LayerType.Softmax];

    return options.map(v => ({ value: v, text: toLayerName(v) }));
}