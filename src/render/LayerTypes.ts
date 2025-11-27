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

export interface IOption<TValue> {
    value: TValue;
    text: string;
}

export const getLayerOptions = (): IOption<LayerType>[] => {
    const options = [LayerType.Linear, LayerType.ReLu, LayerType.Flatten, LayerType.Softmax];

    return options.map(v => ({ value: v, text: toLayerName(v) }));
}