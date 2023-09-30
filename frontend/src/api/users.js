import {get,post} from './http'

export const login=(user)=>post('/users/login',user)

export const getUsernameById=(userId)=>get('/users/username',userId)