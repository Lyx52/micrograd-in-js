import trainLabelsUrl from './train-labels.bin?url';
import trainImagesUrl from './train-images.bin?url';

const fetchBytes = async (url: string) => {
    const res = await fetch(url);
    return await res.bytes()
}

export const trainLabels = async () => await fetchBytes(trainLabelsUrl);
export const trainImages = async () => await fetchBytes(trainImagesUrl);

