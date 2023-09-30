import VueRouter from "vue-router"
//引用组件
import Login from '../pages/Login'
import Home from "@/components/Home";
import HomeInfo from "@/pages/HomeInfo";
import Pictures from "@/pages/Pictures";
import PicturesTable from "@/pages/PicturesTable";

export default new VueRouter({
    routes:[
        {
            path:'/',
            component:Login
        },
        {
            path:'/Home',
            component: Home,
            children:[
                {
                    name:'Pictures',
                    path:'/Pictures',
                    component:Pictures,
                    children:[
                        {
                            name:'PicturesTable',
                            path:'pictures',
                            component:PicturesTable
                        }
                    ]
                },
                {
                    name:'HomeInfo',
                    path:'/HomeInfo',
                    component:HomeInfo
                },
            ]
        }
    ]
})
