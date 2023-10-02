import {get, post, put} from './http'

export const getMyPictures=(pageParam)=>get('/pictures/',pageParam)
export const uploadPictures=(form)=>get('/pictures/upload', form)
export const likePicture=(pictureId)=>put('/pictures/like',pictureId)
export const editPicture=(img)=>post('/picture/edit',img)
export const deletePicture=(pictureId)=>post('/picture/delete',pictureId)